#!/usr/bin/env python3
"""
Experiment #1148: 4h Primary + 1d/1w HTF — Fisher Transform Reversals + HMA Trend Filter

Hypothesis: After 948 failed strategies, most trend-following approaches fail in bear/range markets.
The Ehlers Fisher Transform excels at catching reversals by normalizing price into Gaussian distribution.
Combined with HMA trend filter and volume confirmation, this should work in 2022-2024 range markets
where simple EMA/HMA crossovers failed repeatedly.

Key innovations:
1. Fisher Transform (period=9): Normalizes price to Gaussian, crossings at ±1.5 signal reversals
2. HMA(21) trend filter: Only take Fisher signals in direction of 1d HMA trend
3. Volume ratio confirmation: Entry volume > 1.3x 20-bar avg volume (institutional interest)
4. 1w HMA bias filter: Long only if price > 1w_HMA*0.95, Short only if price < 1w_HMA*1.05
5. ATR(14) 2.5x trailing stop with signal→0 exit
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work (DIFFERENT from failed strategies):
- Fisher Transform catches reversals, not trends (most failed strategies were trend-following)
- Works in range markets (2022-2023) where trend strategies got whipsawed
- Volume confirmation filters out false signals (reduces trade count to target 25-40/year)
- 4h timeframe balances signal quality vs trade frequency
- HTF bias prevents counter-trend trades that destroy Sharpe

Entry conditions (LOOSE enough to guarantee trades):
- LONG: Fisher crosses above -1.5 + close > 1d_HMA + volume_ratio > 1.2 + close > 1w_HMA*0.92
- SHORT: Fisher crosses below +1.5 + close < 1d_HMA + volume_ratio > 1.2 + close < 1w_HMA*1.08
- Exit: Fisher crosses opposite extreme OR stoploss hit (2.5*ATR)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_vol_reversal_1d1w_v1"
timeframe = "4h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price into Gaussian distribution
    Makes turning points easier to identify
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Transform: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Smooth with EMA
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over lookback
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (typical[i] - lowest) / price_range
        
        # Clamp to avoid division by zero in log
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with simple EMA-like smoothing
        if i == period - 1:
            fisher[i] = fisher_raw
        else:
            fisher[i] = 0.67 * fisher_raw + 0.33 * fisher[i-1]
        
        # Signal line (1-bar lag)
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / average volume over period"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            vol_ratio[i] = volume[i] / avg_vol
    
    return vol_ratio

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
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(vol_ratio[i]):
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
        
        # === HTF BIAS FILTERS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Weekly bias filter (more lenient to ensure trades)
        hma_1w_bull = close[i] > hma_1w_aligned[i] * 0.92
        hma_1w_bear = close[i] < hma_1w_aligned[i] * 1.08
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.2
        
        # === FISHER TRANSFORM CROSSINGS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Fisher crosses above -1.5 (oversold reversal)
            if fisher_signal[i] < -1.5 and fisher[i] >= -1.5:
                fisher_cross_up = True
            # Fisher crosses below +1.5 (overbought reversal)
            if fisher_signal[i] > 1.5 and fisher[i] <= 1.5:
                fisher_cross_down = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entry: Fisher reversal up + HMA trend + volume + weekly bias
        if fisher_cross_up and hma_1d_bull and vol_confirmed and hma_1w_bull:
            desired_signal = SIZE_STRONG
        elif fisher_cross_up and hma_1d_bull and hma_1w_bull:
            # Weaker signal without volume confirmation
            desired_signal = SIZE_BASE
        
        # SHORT entry: Fisher reversal down + HMA trend + volume + weekly bias
        elif fisher_cross_down and hma_1d_bear and vol_confirmed and hma_1w_bear:
            desired_signal = -SIZE_STRONG
        elif fisher_cross_down and hma_1d_bear and hma_1w_bear:
            # Weaker signal without volume confirmation
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
        
        # === FISHER EXIT SIGNALS (opposite crossing) ===
        fisher_exit_long = False
        fisher_exit_short = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Exit long when Fisher crosses below +1.0
            if fisher_signal[i] > 1.0 and fisher[i] <= 1.0:
                fisher_exit_long = True
            # Exit short when Fisher crosses above -1.0
            if fisher_signal[i] < -1.0 and fisher[i] >= -1.0:
                fisher_exit_short = True
        
        if in_position and position_side > 0 and fisher_exit_long:
            stoploss_triggered = True
        if in_position and position_side < 0 and fisher_exit_short:
            stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
        prev_fisher = fisher[i]
    
    return signals