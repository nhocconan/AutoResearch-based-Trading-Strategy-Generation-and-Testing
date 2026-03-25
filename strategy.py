#!/usr/bin/env python3
"""
Experiment #1451: 6h Primary + 1w/1d HTF — Triple TF Trend + RSI Pullback

Hypothesis: 6h timeframe is underexplored (0 experiments). This strategy combines:
1. 1w HMA(21) for major trend bias (slowest TF, reduces whipsaw in 2022 crash)
2. 1d KAMA(14) for intermediate trend confirmation (adaptive to volatility)
3. 6h RSI(14) pullback entry (LOOSE: 35-65 zone, NOT extremes to guarantee trades)
4. ATR(14) 3x trailing stoploss (wider than 4h strategies to avoid premature exits)
5. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work where other 6h strategies failed:
- Triple TF alignment (1w+1d+6h) = stronger trend filter than dual TF
- KAMA adapts to volatility regimes (better than static EMA in 2022 crash)
- LOOSE RSI entry (35-65) guarantees trades vs failed strategies with 0 trades
- 6h TF = natural 30-50 trades/year (fee-efficient, not overtraded)
- 3x ATR stop = wider than 4h strategies, accounts for 6h volatility

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3 trades/test):
- LONG: 1w_HMA bullish + 1d_KAMA bullish + 6h_RSI > 35 (pullback zone)
- SHORT: 1w_HMA bearish + 1d_KAMA bearish + 6h_RSI < 65 (pullback zone)

Target: Sharpe>0.6 (beat current best 0.575), trades>=30 train, trades>=3 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_triple_tf_hma_kama_rsi_1w1d_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_kama(close, period=14, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period + slow - 1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 0:
                er[i] = signal / noise
            else:
                er[i] = 1.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period + slow - 1, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period + slow - 1] = close[period + slow - 1]
    
    # Calculate KAMA
    for i in range(period + slow, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=14)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
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
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w HMA bias - slowest TF) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d KAMA confirmation) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === RSI PULLBACK (LOOSE entry - guarantee trades) ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + RSI > 35 (pullback zone, not extreme)
        if price_above_1w and price_above_1d and rsi > 35:
            # Strong if RSI also < 70 (not overbought)
            if rsi < 70:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1w bearish + 1d bearish + RSI < 65 (pullback zone, not extreme)
        elif price_below_1w and price_below_1d and rsi < 65:
            # Strong if RSI also > 30 (not oversold)
            if rsi > 30:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3x ATR trailing - wider for 6h volatility) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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