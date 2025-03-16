const os = require('os');
const { execSync } = require('child_process');
const edge = require('edge-js');
const express = require('express');
const app = express();
app.use(express.json());
const crypto= require('crypto');
// 加载共享库
const prefetchData = edge.func({
    assemblyFile: 'native_extension.dll',
    typeName: 'NativeExtension',
    methodName: 'PrefetchData'
});

const writeDataWithNonTemporalStore = edge.func({
    assemblyFile: 'native_extension.dll',
    typeName: 'NativeExtension',
    methodName: 'WriteDataWithNonTemporalStore'
});
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// 工作窃取调度器
function workStealingScheduler(tasks) {
    const numWorkers = os.cpus().length;
    const workers = [];

    for (let i = 0; i < numWorkers; i++) {
        workers.push(new Worker(__filename, { workerData: { tasks } }));
    }

    workers.forEach(worker => {
        worker.on('message', (message) => {
            if (message === 'steal_task') {
                const task = tasks.pop();
                if (task) {
                    worker.postMessage({ cmd: 'execute_task', task });
                }
            }
        });

        worker.on('exit', (code) => {
            if (code !== 0) {
                console.error(`Worker stopped with exit code ${code}`);
            }
        });
    });

    function stealTask() {
        workers.forEach(worker => worker.postMessage('steal_task'));
    }

    function taskComplete() {
        stealTask();
    }

    function splitTask(task) {
        if (task.complexity > threshold) {
            const subTasks = splitIntoSubTasks(task);
            tasks.push(...subTasks);
            stealTask();
        }
    }

    return { stealTask, taskComplete, splitTask };
}
// 获取系统的CPU核心数
const numCPUs = os.cpus().length;
// 禁用超线程并独占L1/L2缓存
function disableHyperThreadingAndIsolateCache() {
    for (let i = 0; i < numCPUs; i += 2) {
        try {
            execSync(`echo 0 > /sys/devices/system/cpu/cpu${i + 1}/online`);
            execSync(`echo 1 > /sys/devices/system/cpu/cpu${i}/cache/index0/shared_cpu_map`);
            execSync(`echo 1 > /sys/devices/system/cpu/cpu${i}/cache/index1/shared_cpu_map`);
        } catch (error) {
            console.error(`Failed to disable hyperthreading or isolate cache on CPU ${i}:`, error);
        }
    }
}

// 使用PREFETCHT0指令进行缓存预取控制
function prefetchData(address) {
    prefetchData(address, function (error, result) {
        if (error) throw error;
    });
}

// 使用MOVNT指令进行写合并优化
function writeDataWithNonTemporalStore(address, data) {
    writeDataWithNonTemporalStore({ address: address, data: data }, function (error, result) {
        if (error) throw error;
    });
}
// 调整单线程并行度
function adjustThroughput() {
    const instRetired = parseInt(execSync('perf stat -e instructions:u -a sleep 1').toString().match(/(\d+)/)[0]);
    const cpuCycles = parseInt(execSync('perf stat -e cycles:u -a sleep 1').toString().match(/(\d+)/)[0]);
    const ipc = instRetired / cpuCycles;

    const OPTIMAL_IPC = 3.2;  // Skylake架构理论IPC上限

    if (ipc < OPTIMAL_IPC * 0.9) {
        // 增加指令级并行度
        bindThreadToCore(0);  // 绑定至物理核心0
        disableHyperThreadingAndIsolateCache();
    } else {
        // 维持当前发射策略
        execSync('echo 1 > /sys/devices/system/cpu/cpu0/cpufreq/ondemand/up_threshold');
    }
}

// 禁用超线程并独占L1/L2缓存
disableHyperThreadingAndIsolateCache();

// 将线程绑定到特定的物理核心
function bindThreadToCore(coreId) {
    try {
        execSync(`taskset -cp ${coreId} ${process.pid}`);
        console.log(`Thread bound to core ${coreId}`);
    } catch (error) {
        console.error(`Failed to bind thread to core ${coreId}:`, error);
    }
}


// 绑定主线程到物理核心0
bindThreadToCore(0);

// 调整单线程并行度
adjustThroughput();
const originalMathSin = Math.sin;

Math.sin = function(value) {
    const key = `sin_${value}`;
    return autoGetOrComputeValue(key, () => originalMathSin(value));
};
Math.cos = function(value) {
    const key = `cos_${value}`;
    return autoGetOrComputeValue(key, () => originalMathSin(value));
};
Math.tan = function(value) {
    const key = `tan_${value}`;
    return autoGetOrComputeValue(key, () => originalMathSin(value));
};
//hash_table 拦截替换



// 预计算缓存
function precomputeCache() {
    let hashresult;
    // 预计算一些常用的函数值，例如三角函数和哈希表
    for (let i = 0; i < 360; i++) {
        precomputedCache[`sin_${i}`] = Math.sin(i * Math.PI / 180);
        precomputedCache[`cos_${i}`] = Math.cos(i * Math.PI / 180);
        precomputedCache[`tan_${i}`] = Math.tan(i * Math.PI / 180);
        
    }
    for(let i=0;i<10000;i++){
    hashresult[`hash_${type}_${i}`]=hashFunction(type,i);//hash_table
    }
    console.log('Precomputed cache initialized');
}
function hashFunction(type,value){
    let hash = crypto.createHash(type, value);
    return hash
}


// 初始化预计算缓存
precomputeCache();
// 圆锥曲线计算
function conicSectionCalculation(a, b, c, d, e, f) {
    const delta = b * b - 4 * a * c;
    let type, parameters;

    if (delta > 0) {
        type = 'Hyperbola';
        parameters = calculateHyperbolaParameters(a, b, c, d, e, f);
    } else if (delta === 0) {
        type = 'Parabola';
        parameters = calculateParabolaParameters(a, b, c, d, e, f);
    } else {
        type = 'Ellipse';
        parameters = calculateEllipseParameters(a, b, c, d, e, f);
    }

    return { type, parameters };
}
function calculateHyperbolaParameters(a, b, c, d, e, f) {
    // 计算双曲线的参数，如焦点、准线、离心率等
    const h = -d / (2 * a);
    const k = -e / (2 * c);
    const a2 = Math.abs(1 / a);
    const b2 = Math.abs(1 / c);
    const eccentricity = Math.sqrt(1 + b2 / a2);
    const fociDistance = 2 * Math.sqrt(a2 + b2);
    return {
        center: [h, k],
        semiMajorAxis: Math.sqrt(a2),
        semiMinorAxis: Math.sqrt(b2),
        eccentricity: eccentricity,
        foci: [[h + fociDistance / 2, k], [h - fociDistance / 2, k]]
    };
}

function calculateParabolaParameters(a, b, c, d, e, f) {
    // 计算抛物线的参数，如焦点、准线、离心率等
    const h = -d / (2 * a);
    const k = -e / (2 * c);
    const p = 1 / (4 * a);
    return {
        vertex: [h, k],
        focus: [h, k + p],
        directrix: `y = ${k - p}`,
        eccentricity: 1
    };
}

function calculateEllipseParameters(a, b, c, d, e, f) {
    // 计算椭圆的参数，如焦点、准线、离心率等
    const h = -d / (2 * a);
    const k = -e / (2 * c);
    const a2 = Math.abs(1 / a);
    const b2 = Math.abs(1 / c);
    const eccentricity = Math.sqrt(1 - b2 / a2);
    const fociDistance = 2 * Math.sqrt(a2 - b2);
    return {
        center: [h, k],
        semiMajorAxis: Math.sqrt(a2),
        semiMinorAxis: Math.sqrt(b2),
        eccentricity: eccentricity,
        foci: [[h + fociDistance / 2, k], [h - fociDistance / 2, k]]
    };
}
// 矩阵运算
function matrixOperation(matrixA, matrixB, operation) {
    const result = [];
    for (let i = 0; i < matrixA.length; i++) {
        result[i] = [];
        for (let j = 0; j < matrixA[i].length; j++) {
            if (operation === 'add') {
                result[i][j] = matrixA[i][j] + matrixB[i][j];
            } else if (operation === 'subtract') {
                result[i][j] = matrixA[i][j] - matrixB[i][j];
            } else if (operation === 'multiply') {
                result[i][j] = matrixA[i][j] * matrixB[i][j];
            } else if (operation === 'divide') {
                result[i][j] = matrixA[i][j] / matrixB[i][j];
            }
        }
    }
    return result;
}

// 流体力学方程计算
function fluidDynamicsEquationCalculation(equation, parameters) {
    if (equation === 'Navier-Stokes') {
        return calculateNavierStokes(parameters);
    } else if (equation === 'Euler') {
        return calculateEuler(parameters);
    } else if (equation === 'Bernoulli') {
        return calculateBernoulli(parameters);
    } else {
        return 'Unknown Equation';
    }
}

function calculateNavierStokes({ u, v, p, rho, mu, dx, dy, dt }) {
    // 计算Navier-Stokes方程
    const du_dt = -u * (u / dx) - v * (u / dy) - (1 / rho) * (p / dx) + mu * ((u / (dx * dx)) + (u / (dy * dy)));
    const dv_dt = -u * (v / dx) - v * (v / dy) - (1 / rho) * (p / dy) + mu * ((v / (dx * dx)) + (v / (dy * dy)));
    return { du_dt: du_dt.toFixed(10), dv_dt: dv_dt.toFixed(10) };
}

function calculateEuler({ u, v, p, rho, dx, dy }) {
    // 计算Euler方程
    const du_dt = -u * (u / dx) - v * (u / dy) - (1 / rho) * (p / dx);
    const dv_dt = -u * (v / dx) - v * (v / dy) - (1 / rho) * (p / dy);
    return { du_dt: du_dt.toFixed(10), dv_dt: dv_dt.toFixed(10) };
}

function calculateBernoulli({ p1, v1, p2, v2, rho }) {
    // 计算Bernoulli方程
    const h1 = p1 / (rho * 9.81) + (v1 * v1) / (2 * 9.81);
    const h2 = p2 / (rho * 9.81) + (v2 * v2) / (2 * 9.81);
    return { h1: h1.toFixed(10), h2: h2.toFixed(10) };
}
// 获取系统信息
function getSystemInfo() {
    const cpuCores = os.cpus().length;
    const totalMemory = `${Math.round(os.totalmem() / 1024 / 1024 / 1024)}GB`;
    const availableMemory = `${Math.round(os.freemem() / 1024 / 1024 / 1024)}GB`;
    return { cpu_cores: cpuCores, total_memory: totalMemory, available_memory: availableMemory };
}
let tasks = {};

// 获取任务状态
function getTaskStatus(taskId) {
    const task = tasks[taskId];
    if (task) {
        return { task_id: taskId, ...task };
    } else {
        return { error: 'Task not found' };
    }
}
// 公式转换：矩阵分块
function matrixBlockConversion(matrix, blockSize) {
    const blocks = [];
    for (let i = 0; i < matrix.length; i += blockSize) {
        for (let j = 0; j < matrix[i].length; j += blockSize) {
            const block = [];
            for (let k = 0; k < blockSize; k++) {
                block.push(matrix[i + k].slice(j, j + blockSize));
            }
            blocks.push(block);
        }
    }
    return blocks;
}

// 公式转换：FFT蝶形运算
function fftButterflyOperation(input) {
    const n = input.length;
    if (n <= 1) return input;

    const even = fftButterflyOperation(input.filter((_, i) => i % 2 === 0));
    const odd = fftButterflyOperation(input.filter((_, i) => i % 2 !== 0));

    const result = new Array(n).fill(0);
    for (let k = 0; k < n / 2; k++) {
        const t = Math.exp(-2 * Math.PI * k / n) * odd[k];
        result[k] = even[k] + t;
        result[k + n / 2] = even[k] - t;
    }
    return result;
}

// 动态规划：构建状态转移表
function dynamicProgrammingTable(n) {
    const dp = new Array(n + 1).fill(0);
    dp[0] = 1;
    for (let i = 1; i <= n; i++) {
        for (let j = i; j <= n; j++) {
            dp[j] += dp[j - i];
        }
    }
    return dp;
}
function autoGetOrComputeValue(key, computeFunc) {
    if (precomputedCache.hasOwnProperty(key)) {
        console.log(`Cache hit for key: ${key}`);
        return precomputedCache[key];
    } else {
        console.log(`Cache miss for key: ${key}, computing value...`);
        const value = computeFunc();
        precomputedCache[key] = value;
        return value;
    }
}
if (!isMainThread) {
    parentPort.on('message', (message) => {
        if (message.cmd === 'execute_task') {
            const task = message.task;
            // 执行任务
            const result = task.execute();
            parentPort.postMessage({ cmd: 'task_complete', result });
        }
    });

    parentPort.postMessage('steal_task');
}
// 任务拆分：多项式计算
function splitPolynomialTask(coefficients) {
    return coefficients.map((coef, index) => ({
        complexity: 1,
        execute: () => coef * Math.pow(index, index)
    }));
}
function computematrixTask(matrixA, matrixB,operation) {
    for (let i = 0; i < matrixA.length; i++) {
        for (let j = 0; j < matrixA[i].length; j++) {
            tasks.push({
                complexity: 1,
                execute: () => {
                    if (operation === 'add') {
                        return matrixA[i][j] + matrixB[i][j];
                    } else if (operation === 'subtract') {
                        return matrixA[i][j] - matrixB[i][j];
                    } else if (operation === 'multiply') {
                        return matrixA[i][j] * matrixB[i][j];
                    } else if (operation === 'divide') {
                        return matrixA[i][j] / matrixB[i][j];
                    }
                }
            });
        }
    }
    return tasks;
}
// 任务拆分：矩阵计算
function splitMatrixTask(matrixA, matrixB, operation) {
    const tasks = [];
    const blockSize = 4;
    const blockMatrixA = matrixBlockConversion(matrixA, blockSize);
    const blockMatrixB = matrixBlockConversion(matrixB, blockSize);
    for (let i = 0; i < blockMatrixA.length; i++) {
        for(let j=0; j<blockMatrixB.length; j++) {
            computematrixTask(blockMatrixA[i], blockMatrixB[j],operation);
        }

    }

}

// 任务拆分：流体力学方程计算
function splitFluidDynamicsTask(equation, parameters) {
    // 根据具体的流体力学方程拆分任务
    // 这里只是一个示例，具体实现需要根据方程的复杂度进行拆分
    return [{
        complexity: 1,
        execute: () => fluidDynamicsEquationCalculation(equation, parameters)
    }];
}
// 自动加速任务
function autoAccelerate(taskType, parameters) {
    let tasks;

    if (taskType === 'polynomial') {
        const { coefficients } = parameters;
        tasks = splitPolynomialTask(coefficients);
    } else if (taskType === 'matrix_operation') {
        const { matrixA, matrixB, operation } = parameters;
        tasks = splitMatrixTask(matrixA, matrixB, operation);

        // 调用预取和写合并优化
        for (let i = 0; i < matrixA.length; i++) {
            for (let j = 0; j < matrixA[i].length; j++) {
                prefetchData(matrixA[i][j]);
                writeDataWithNonTemporalStore(matrixB[i][j], matrixA[i][j]);
            }
        }
    } else if (taskType === 'fluid_dynamics') {
        const { equation } = parameters;
        tasks = splitFluidDynamicsTask(equation, parameters);
    } else {
        throw new Error('Invalid task type');
    }

    const scheduler = workStealingScheduler(tasks);
    scheduler.stealTask();
}

// 示例计算任务
function computeTask(taskType, parameters) {
    let result;

    if (taskType === 'conic_section') {
        const { a, b, c, d, e, f } = parameters;
        result = conicSectionCalculation(a, b, c, d, e, f);
    } else if (taskType === 'matrix_operation') {
        const { matrixA, matrixB, operation } = parameters;
        result = matrixOperation(matrixA, matrixB, operation);

        // 调用预取和写合并优化
        for (let i = 0; i < matrixA.length; i++) {
            for (let j = 0; j < matrixA[i].length; j++) {
                prefetchData(matrixA[i][j]);
                writeDataWithNonTemporalStore(matrixB[i][j], matrixA[i][j]);
            }
        }
    } else if (taskType === 'fluid_dynamics') {
        const { equation } = parameters;
        result = fluidDynamicsEquationCalculation(equation, parameters);
    } else if (taskType === 'matrix_block_conversion') {
        const { matrix, blockSize } = parameters;
        result = matrixBlockConversion(matrix, blockSize);
    } else if (taskType === 'fft_butterfly_operation') {
        const { input } = parameters;
        result = fftButterflyOperation(input);
    } else if (taskType === 'dynamic_programming') {
        const { n } = parameters;
        result = dynamicProgrammingTable(n);
    } else {
        throw new Error('Invalid task type');
    }

    return result;
}
// main_api 函数
function main_api() {
    app.get('/api/system_info', (req, res) => {
        const systemInfo = getSystemInfo();
        res.json(systemInfo);
    });

    app.post('/api/compute', (req, res) => {
        const { task_type, parameters } = req.body;
        let result;

        try {
            result = computeTask(task_type, parameters);
            autoAccelerate(task_type, parameters);
        } catch (error) {
            res.status(400).json({ error: error.message });
            return;
        }

        const taskId = Date.now().toString();
        tasks[taskId] = { status: 'success', result };
        res.json({ task_id: taskId, status: 'running' });
    });

    app.get('/api/task_status', (req, res) => {
        const { task_id } = req.query;
        const taskStatus = getTaskStatus(task_id);
        res.json(taskStatus);
    });

    app.listen(3000, () => {
        console.log('API server is running on port 3000');
    });
}

// main_cli 函数
function main_cli(...args) {
    const command = args[0];
    let result;

    if (command === 'system_info') {
        result = getSystemInfo();
    } else if (command === 'compute_task') {
        const taskType = args[2];
        const parameters = JSON.parse(args[4]);
        try {
            result = computeTask(taskType, parameters);
            autoAccelerate(taskType, parameters);
        } catch (error) {
            console.error('Error:', error.message);
            return;
        }

        const taskId = Date.now().toString();
        tasks[taskId] = { status: 'success', result };
        console.log(`Task ID: ${taskId}`);
        console.log(`Status: success`);
        console.log(`Result: ${JSON.stringify(result)}`);
    } else if (command === 'task_status') {
        const taskId = args[2];
        result = getTaskStatus(taskId);
    } else {
        console.error('Invalid command');
        return;
    }

    console.log('Result:', result);
}

// 根据命令行参数执行相应的函数
const args = process.argv.slice(2);
if (args.length > 0) {
    main_cli(...args);
} else {
    main_api();
}