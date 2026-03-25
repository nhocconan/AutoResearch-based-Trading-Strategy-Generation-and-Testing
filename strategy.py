#!/usr/bin/env python3
"""
Experiment #1362: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Transform Reversals

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adjusts to market noise better than EMA/HMA,
reducing whipsaws in choppy conditions. Combined with Ehlers Fisher Transform for precise
reversal entries, this should capture trend continuations with better timing than simple crossovers.

Why this should work where others failed:
1. KAMA adapts ER (Efficiency Ratio) - slows in chop, speeds in trends
2. Fisher Transform normalizes price to Gaussian distribution, extreme values (-2/+2) mark reversals
3. 1d KAMA for major trend bias avoids counter-trend trades
4. Volume confirmation filters false breakouts
5. 4h TF = natural 25-40 trades/year (fee-friendly, not too sparse)

Entry logic:
- LONG: 1d_KAMA sloping up + 4h_Fisher crosses above -1.5 + volume > 20-bar avg
- SHORT: 1d_KAMA sloping down + 4h_Fisher crosses below +1.5 + volume > 20-bar avg

Exit: 2.5x ATR trailing stop OR Fisher crosses opposite extreme

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_volume_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio: net change / sum of absolute changes
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]):
            net_change = abs(close[i] - close[i - period])
            sum_changes = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j-1]):
                    sum_changes += abs(close[j] - close[j-1])
            if sum_changes > 1e-10:
                er[i] = net_change / sum_changes
    
    # Smoothing constant
    sc = np.zeros(n, dtype=np.float64)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(close[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate typical price and normalize
    for i in range(period - 1, n):
        # Highest high and lowest low over period
        hh = np.nanmax(high[i - period + 1:i + 1])
        ll = np.nanmin(low[i - period + 1:i + 1])
        
        if hh > ll:
            # Normalize to 0-1 range
            x = (2.0 * ((high[i] + low[i]) / 2.0 - ll) / (hh - ll)) - 1.0
            # Clamp to avoid division issues
            x = max(-0.999, min(0.999, x))
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
            if i > 0 and not np.isnan(fisher[i-1]):
                fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        vol_window = volume[i - period + 1:i + 1]
        if not np.any(np.isnan(vol_window)):
            vol_sma[i] = np.mean(vol_window)
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=10)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
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
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_4h[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA slope) ===
        # Check if 1d KAMA is sloping up or down
        kama_1d_slope = 0
        if i > 4 and not np.isnan(kama_1d_aligned[i-4]):
            if kama_1d_aligned[i] > kama_1d_aligned[i-4]:
                kama_1d_slope = 1  # Up
            elif kama_1d_aligned[i] < kama_1d_aligned[i-4]:
                kama_1d_slope = -1  # Down
        
        # 1w KAMA for major regime
        price_above_1w = close[i] > kama_1w_aligned[i] if not np.isnan(kama_1w_aligned[i]) else False
        price_below_1w = close[i] < kama_1w_aligned[i] if not np.isnan(kama_1w_aligned[i]) else False
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i]
        
        # Fisher crossing above -1.5 (bullish reversal from oversold)
        fisher_bull_cross = (fisher_prev_val < -1.5) and (fisher_val >= -1.5)
        # Fisher crossing below +1.5 (bearish reversal from overbought)
        fisher_bear_cross = (fisher_prev_val > 1.5) and (fisher_val <= 1.5)
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > vol_sma_20[i] if not np.isnan(vol_sma_20[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1d KAMA up + Fisher bull cross + volume confirmation
        if kama_1d_slope == 1 and fisher_bull_cross and volume_ok:
            if price_above_1w:
                desired_signal = SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = SIZE_BASE  # Basic long
        
        # SHORT: 1d KAMA down + Fisher bear cross + volume confirmation
        elif kama_1d_slope == -1 and fisher_bear_cross and volume_ok:
            if price_below_1w:
                desired_signal = -SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = -SIZE_BASE  # Basic short
        
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
        
        # Fisher exit signal (opposite cross)
        fisher_exit = False
        if in_position and position_side > 0 and fisher_bear_cross:
            fisher_exit = True
        if in_position and position_side < 0 and fisher_bull_cross:
            fisher_exit = True
        
        if stoploss_triggered or fisher_exit:
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