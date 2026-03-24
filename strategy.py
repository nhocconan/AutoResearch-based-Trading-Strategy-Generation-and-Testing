#!/usr/bin/env python3
"""
Experiment #1571: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend Strategy

Hypothesis: After 11 failed 4h experiments with complex regime switching, simpler is better.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - follows price in trends,
flattens in ranges. This should reduce whipsaw compared to HMA/EMA crossovers.

Key innovations:
1. KAMA(10,2,30) on 4h - adapts to volatility, less lag than EMA
2. 1d HMA(21) for trend bias (proven in best strategies)
3. 1w HMA(21) for long-term regime filter (only trade with weekly trend)
4. RSI(14) 45-55 filter for trend continuation (not extreme reversals)
5. ATR(14) 2.5x trailing stop for drawdown control
6. Discrete position sizing (0.30) to minimize fee churn

Why this should beat Sharpe 0.618:
- KAMA reduces whipsaw in choppy 4h markets (major issue in prior experiments)
- Dual HTF filter (1d + 1w) ensures we only trade with major trend
- RSI continuation filter (45-55) catches trend continuations, not reversals
- Simpler logic = more reliable signals = more trades (>30/year target)
- 4h targets 20-50 trades/year — optimal for fee efficiency

Timeframe: 4h (required for this experiment)
HTF: 1d HMA + 1w HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_trend_1d1w_hma_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=10, slow_period=30, smooth_period=2):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency ratio (ER)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    """
    n = len(close)
    if n < slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(slow_period - 1, n):
        signal = abs(close[i] - close[i - slow_period + 1])
        noise = np.sum(np.abs(np.diff(close[i - slow_period + 1:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA at SMA of first slow_period bars
    kama[slow_period - 1] = np.mean(close[:slow_period])
    
    # Calculate KAMA
    for i in range(slow_period, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # KAMA adaptive moving averages
    kama_fast = calculate_kama(close, fast_period=10, slow_period=30, smooth_period=2)
    kama_slow = calculate_kama(close, fast_period=20, slow_period=50, smooth_period=2)
    
    # HMA for additional confirmation
    hma_4h = calculate_hma(close, period=21)
    
    # SMA 200 for long-term trend
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA + 1w HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND SIGNAL ===
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # === RSI CONTINUATION FILTER (45-55 for trend continuation) ===
        rsi_neutral = 45.0 <= rsi[i] <= 55.0
        rsi_bull = rsi[i] >= 45.0
        rsi_bear = rsi[i] <= 55.0
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: KAMA bull + Daily bull + Weekly bull (or neutral) + RSI support
        if kama_bull and daily_bull and (weekly_bull or not weekly_bear) and rsi_bull:
            desired_signal = BASE_SIZE
        
        # SHORT: KAMA bear + Daily bear + Weekly bear (or neutral) + RSI support
        elif kama_bear and daily_bear and (weekly_bear or not weekly_bull) and rsi_bear:
            desired_signal = -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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