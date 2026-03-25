#!/usr/bin/env python3
"""
Experiment #1540: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Transform Entries

Hypothesis: 6h timeframe sits in the "sweet spot" between 4h (too many trades) and 12h (too few).
This strategy uses ADAPTIVE indicators that adjust to market volatility:

1. KAMA (Kaufman Adaptive Moving Average): Adjusts smoothing based on market efficiency.
   Fast in trends (low lag), slow in chop (filters noise). Perfect for crypto's regime shifts.
2. Ehlers Fisher Transform: Normalizes price to -1 to +1 range. Excellent for spotting reversals
   at extremes. Entry when Fisher crosses -0.8 (long) or +0.8 (short).
3. 1d HMA(21) for intermediate trend bias
4. 1w HMA(21) for ultra-long-term bias (only trade with weekly trend)
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work on 6h:
- KAMA adapts to BTC/ETH's varying volatility regimes (2021 bull, 2022 crash, 2023-24 range)
- Fisher Transform catches reversals that RSI misses (proven in Ehlers literature)
- 1w HTF filter prevents major counter-trend positions (critical for 2022 crash survival)
- 6h TF = natural 30-50 trades/year (fee-efficient, meets minimum trade requirements)
- LOOSE Fisher thresholds (-0.8/+0.8, not -1.0/+1.0) guarantee sufficient trades

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + 1d_HMA bullish + KAMA rising + Fisher crosses above -0.8
- SHORT: 1w_HMA bearish + 1d_HMA bearish + KAMA falling + Fisher crosses below +0.8

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_fisher_1d1w_adaptive_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in choppy markets.
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        if price_change < 1e-10:
            er[i] = 0.0
        else:
            vol_sum = np.sum(np.abs(np.diff(close[i - period:i + 1])))
            if vol_sum > 1e-10:
                er[i] = price_change / vol_sum
            else:
                er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    sc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to range -1 to +1 for clearer reversal signals.
    Entry when Fisher crosses -0.8 (long) or +0.8 (short).
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2 - lowest_low) / price_range
        
        # Clamp to avoid division by zero
        normalized = max(0.001, min(0.999, normalized))
        
        # Calculate intermediate value
        temp = 0.66 * ((normalized - 0.5) + 0.66 * (normalized - 0.5) if i > period - 1 else 0)
        
        # Use simpler Fisher calculation
        x = (2 * normalized - 1)
        x = max(-0.999, min(0.999, x))
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Fisher signal (previous bar's fisher)
        if i > period - 1:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_20 = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
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
    min_bars = 100
    
    # Track Fisher crosses to avoid repeated signals
    prev_fisher_long_signal = False
    prev_fisher_short_signal = False
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(kama_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
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
        
        # === TREND DIRECTION (1w and 1d HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (adaptive momentum) ===
        kama_bullish = kama_10[i] > kama_20[i]
        kama_bearish = kama_10[i] < kama_20[i]
        
        # KAMA direction
        kama_rising = False
        kama_falling = False
        if i > 0 and not np.isnan(kama_10[i-1]):
            kama_rising = kama_10[i] > kama_10[i-1]
            kama_falling = kama_10[i] < kama_10[i-1]
        
        # === FISHER TRANSFORM (reversal signals) ===
        fisher_val = fisher[i]
        fisher_prev = fisher_signal[i]
        
        # Fisher cross above -0.8 (oversold reversal → long)
        fisher_long_cross = (fisher_prev < -0.8 and fisher_val >= -0.8) if not np.isnan(fisher_prev) else False
        
        # Fisher cross below +0.8 (overbought reversal → short)
        fisher_short_cross = (fisher_prev > 0.8 and fisher_val <= 0.8) if not np.isnan(fisher_prev) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + KAMA bullish + Fisher long cross
        if price_above_1w and price_above_1d and kama_bullish and fisher_long_cross:
            desired_signal = SIZE_STRONG
        
        # SHORT: 1w bearish + 1d bearish + KAMA bearish + Fisher short cross
        elif price_below_1w and price_below_1d and kama_bearish and fisher_short_cross:
            desired_signal = -SIZE_STRONG
        
        # Secondary entry (looser): Only 1d confirmation + KAMA + Fisher
        elif price_above_1d and kama_bullish and kama_rising and fisher_long_cross:
            desired_signal = SIZE_BASE
        
        elif price_below_1d and kama_bearish and kama_falling and fisher_short_cross:
            desired_signal = -SIZE_BASE
        
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