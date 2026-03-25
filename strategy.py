#!/usr/bin/env python3
"""
Experiment #1420: 6h Primary + 1d/1w HTF — Adaptive KAMA Trend + RSI Entry

Hypothesis: 6h timeframe is unexplored territory (0 prior experiments). This strategy combines:
1. 1w HMA(21) for ultra-long-term bias (avoid major counter-trend trades)
2. 1d HMA(16/48) crossover for medium-term trend confirmation
3. 6h KAMA(14) as primary trend indicator (adaptive to volatility, less lag than EMA)
4. 6h RSI(7) for faster entry timing (catches pullbacks better than RSI14)
5. ATR-based volatility scaling (reduce size when vol spikes, protect drawdown)
6. Trailing stoploss via signal→0 (2.5x ATR)

Why this should beat #1411 (Sharpe=0.575):
- KAMA adapts to market regime (fast in trends, slow in chop) — better than static HMA
- 1w HTF filter prevents major counter-trend positions (critical for 2022 crash)
- RSI(7) is faster than ROC, catches more pullback entries
- Volatility-scaled sizing reduces exposure during high-vol periods (2022, 2025)
- 6h TF = natural 30-50 trades/year (fee-efficient, not overtraded)

Entry logic (LOOSE to guarantee trades):
- LONG: 1w_HMA bullish + 1d_HMA16>48 + 6h_KAMA rising + RSI(7) > 35
- SHORT: 1w_HMA bearish + 1d_HMA16<48 + 6h_KAMA falling + RSI(7) < 65

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete, volatility-scaled
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_adaptive_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    period: lookback for efficiency ratio
    fast: fast SC constant (default 2/16 = 0.125)
    slow: slow SC constant (default 2/32 = 0.0625)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 0:
                er[i] = signal / noise
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_16_raw = calculate_hma(df_1d['close'].values, period=16)
    hma_1d_16_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_16_raw)
    
    hma_1d_48_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_48_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_48_raw)
    
    # Calculate 6h indicators
    kama_14 = calculate_kama(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    # KAMA slope (trend direction)
    kama_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(kama_14[i]) and not np.isnan(kama_14[i-1]):
            kama_slope[i] = kama_14[i] - kama_14[i-1]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(kama_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_16_aligned[i]) or np.isnan(hma_1d_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w HMA bias - ultra long term) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND CONFIRMATION (1d HMA crossover) ===
        hma_1d_bullish = hma_1d_16_aligned[i] > hma_1d_48_aligned[i]
        hma_1d_bearish = hma_1d_16_aligned[i] < hma_1d_48_aligned[i]
        
        # === 6h KAMA TREND (adaptive) ===
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # === RSI ENTRY (fast, catches pullbacks) ===
        rsi = rsi_7[i]
        
        # === VOLATILITY SCALING (reduce size in high vol) ===
        # Use ATR ratio to scale position size
        atr_ratio = atr_14[i] / np.nanmean(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_scale = 1.0
        if atr_ratio > 1.5:
            vol_scale = 0.7  # Reduce size 30% in high vol
        elif atr_ratio > 2.0:
            vol_scale = 0.5  # Reduce size 50% in extreme vol
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d HMA bullish + KAMA rising + RSI > 35 (not extreme)
        if price_above_1w and hma_1d_bullish and kama_rising and rsi > 35:
            if rsi < 70:
                desired_signal = SIZE_STRONG * vol_scale
            else:
                desired_signal = SIZE_BASE * vol_scale
        
        # SHORT: 1w bearish + 1d HMA bearish + KAMA falling + RSI < 65 (not extreme)
        elif price_below_1w and hma_1d_bearish and kama_falling and rsi < 65:
            if rsi > 30:
                desired_signal = -SIZE_STRONG * vol_scale
            else:
                desired_signal = -SIZE_BASE * vol_scale
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals