#!/usr/bin/env python3
"""
Experiment #083: 1d Primary + 1w HTF — HMA Crossover with Weekly Trend Bias

Hypothesis: After 82 failed experiments, the pattern is clear — complex regime filters
destroy performance. Simple trend-following on 1d with 1w bias worked in research
(HMA crossover + RSI filter + ATR trail = SOL Sharpe +0.879).

Key insights from failures:
- TOO MANY FILTERS = 0 trades or negative Sharpe (experiments #071-#082 all failed)
- 1d timeframe naturally gives 20-50 trades/year (fee-efficient)
- 1w HMA provides major trend bias without over-filtering
- Simple HMA crossover generates regular signals in crypto volatility

Strategy Logic:
1. 1w HMA(21) = major trend bias (only long if price > 1w HMA, only short if <)
2. 1d HMA(21/50) crossover = entry signal (fast crosses slow)
3. RSI(14) filter = loose thresholds (30/70) to ensure trades generate
4. ATR(14) 2.5x trailing stop = risk management
5. Discrete sizing: 0.30 for clean entries/exits

Why this should beat current best (Sharpe=0.368):
- Simpler = fewer bugs, more reliable execution
- 1d/1w combination proven in research for ETH/SOL
- Loose RSI ensures we get trades (learned from 0-trade failures)
- Weekly bias prevents counter-trend positions in major moves

Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_crossover_rsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_fast = calculate_hma(close, period=21)
    hma_slow = calculate_hma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === HMA CROSSOVER DETECTION (1d) ===
        # Check if fast crossed above/below slow THIS bar
        crossover_long = (hma_fast[i] > hma_slow[i]) and (hma_fast[i-1] <= hma_slow[i-1])
        crossover_short = (hma_fast[i] < hma_slow[i]) and (hma_fast[i-1] >= hma_slow[i-1])
        
        # === RSI FILTER (LOOSE thresholds to ensure trades) ===
        rsi_ok_long = rsi[i] > 30.0  # Not extremely oversold
        rsi_ok_short = rsi[i] < 70.0  # Not extremely overbought
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: HMA crossover + RSI filter + 1w HMA bullish
        if crossover_long and rsi_ok_long and hma_1w_bull:
            desired_signal = SIZE
        
        # Short entry: HMA crossover + RSI filter + 1w HMA bearish
        elif crossover_short and rsi_ok_short and hma_1w_bear:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals