#!/usr/bin/env python3
"""
Experiment #112: 12h Primary + 1d HTF — HMA Trend + Loose RSI + BB Vol Filter

Hypothesis: After analyzing 95+ failed experiments, the pattern for 12h is clear:
- Complex regime filters (Choppiness, Fisher, dual-regime) = 0 trades (exp #102, #103, #106)
- 12h needs VERY loose entry conditions to generate minimum 10 trades/symbol
- HMA is more responsive than KAMA for 12h (SOL +0.879 with HMA in history)
- RSI thresholds must be 20/80 (not 30/70) to ensure trade generation
- BB width as soft filter (not hard requirement) to avoid whipsaws in low vol

This strategy uses MINIMAL filters for 12h:
1. 1d HMA = major trend bias (price above/below)
2. 12h HMA crossover (8/21) = entry trigger
3. RSI very loose filter (>20 for long, <80 for short) - ensures trades on ALL symbols
4. BB width > 20th percentile (avoid dead markets, but not strict)
5. ATR trailing stoploss (2.5x) for risk management
6. NO Choppiness, NO Fisher, NO complex regime detection

Key design choices:
- Timeframe: 12h (as instructed, target 20-50 trades/year)
- HTF: 1d for trend bias (responsive enough for 12h entries)
- HMA: 8/21 crossover (proven responsive, less lag than EMA)
- RSI thresholds: 20/80 (very loose, ensures trades on BTC/ETH/SOL)
- Position size: 0.30 (30% of capital, conservative for 12h)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_loose_bb_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - more responsive than EMA with less lag
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='valid')
        return result
    
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    if half < 1 or sqrt_period < 1:
        return np.full(n, np.nan)
    
    # Need enough data for all WMA calculations
    min_len = period + half + sqrt_period
    if n < min_len:
        return np.full(n, np.nan)
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # Align arrays (wma_half is shorter by half-1, wma_full by period-1)
    offset = period - half
    if len(wma_half) > offset:
        wma_diff = 2 * wma_half[offset:] - wma_full
    else:
        return np.full(n, np.nan)
    
    hma = wma(wma_diff, sqrt_period)
    
    # Pad with NaN to match original length
    result = np.full(n, np.nan)
    start_idx = period + sqrt_period - 1
    end_idx = start_idx + len(hma)
    if end_idx <= n:
        result[start_idx:end_idx] = hma
    
    return result

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with width calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma
    
    return upper, lower, width

def calculate_percentile_rank(series, lookback=100):
    """Percentile rank of current value vs last lookback values"""
    n = len(series)
    pr = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.isnan(series[i]):
            continue
        window = series[i-lookback+1:i+1]
        window = window[~np.isnan(window)]
        if len(window) > 0:
            pr[i] = np.sum(window < series[i]) / len(window) * 100
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_fast = calculate_hma(close, period=8)
    hma_slow = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate BB width percentile for volatility filter
    bb_width_pr = calculate_percentile_rank(bb_width, lookback=100)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 12h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        # Simple: is price above or below daily HMA?
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === RSI FILTER (VERY LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI > 20 (not extremely oversold, allows many entries)
        # For shorts: RSI < 80 (not extremely overbought, allows many entries)
        rsi_ok_long = rsi[i] > 20.0
        rsi_ok_short = rsi[i] < 80.0
        
        # === BB WIDTH FILTER (Soft - avoid dead markets) ===
        # Only require BB width > 10th percentile (very loose)
        vol_ok = bb_width_pr[i] > 10.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 12h HMA cross bull + RSI > 20 + vol ok
        # SHORT: 1d bear + 12h HMA cross bear + RSI < 80 + vol ok
        desired_signal = 0.0
        
        if htf_bull and hma_cross_bull and rsi_ok_long and vol_ok:
            desired_signal = SIZE
        elif htf_bear and hma_cross_bear and rsi_ok_short and vol_ok:
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