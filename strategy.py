#!/usr/bin/env python3
"""
Experiment #015: 6h Fisher Transform Extremes + Volume Spike + Weekly Trend

HYPOTHESIS: Fisher Transform identifies price extremes where reversals are likely.
In bear markets (2022 crash, 2025 test), catching reversals at extremes outperforms
trend-following which gets whipsawed. Volume spike confirms institutional participation.
Weekly HMA provides dominant trend bias to avoid counter-trend traps.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Fisher extremes capture panic bottoms (bear) and euphoria tops (bull)
- Volume confirmation filters false breakouts common in 2022-2025
- Weekly trend bias prevents dangerous counter-trend entries
- Tight entry conditions (Fisher < -1.8 or > +1.8) ensure few but high-quality trades

TARGET: 50-150 total trades over 4 years (12-37/year). This is MEAN REVERSION so
fewer trades than trend-following, but higher win rate per trade.

KEY DESIGN:
1. Fisher Transform(14) extremes: < -1.8 long, > +1.8 short
2. Volume spike: > 2.0x 20-avg (institutional confirmation)
3. Weekly HMA(21) trend bias: long only if price > weekly HMA, short if <
4. Stoploss: 2.5 ATR trailing
5. Exit: Fisher crosses back through 0 (mean reversion complete)
6. Signal: 0.25 (discrete, conservative sizing)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_extreme_vol_weekly_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(close, period=14):
    """
    Ehlers Fisher Transform - identifies price extremes for reversals
    Normalizes price to -1 to +1 range, then applies Fisher equation
    Values < -1.5 or > +1.5 indicate extreme oversold/overbought
    """
    n = len(close)
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        highest = np.max(window)
        lowest = np.min(window)
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        normalized = 0.999 * (close[i] - lowest) / price_range
        if normalized <= 0.001:
            normalized = 0.001
        if normalized >= 0.999:
            normalized = 0.999
        
        fisher_value = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        fisher[i] = fisher_value
        
        if i >= period + 2:
            fisher_signal[i] = 0.67 * fisher[i] + 0.33 * fisher[i - 1]
        elif i >= period:
            fisher_signal[i] = fisher[i]
    
    return fisher_signal

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

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly data for trend bias (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher = calculate_fisher_transform(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for Fisher + ATR + Volume
    warmup = 80
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]) or vol_ratio[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (Weekly HMA) ===
        price_above_weekly = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 2.0  # Strict: 2x average volume
        
        # === FISHER EXTREMES ===
        fisher_extreme_low = fisher[i] < -1.8  # Very oversold
        fisher_extreme_high = fisher[i] > 1.8   # Very overbought
        
        # === ENTRY LOGIC (Mean Reversion at Extremes) ===
        desired_signal = 0.0
        
        # LONG: Fisher extreme low + volume spike + bullish weekly bias
        if fisher_extreme_low and vol_spike and price_above_weekly:
            desired_signal = SIZE
        
        # SHORT: Fisher extreme high + volume spike + bearish weekly bias
        if fisher_extreme_high and vol_spike and not price_above_weekly:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR Trailing) ===
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
        
        # === TAKE PROFIT (Fisher Mean Reversion Complete) ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # Exit when Fisher crosses back above 0 (mean reversion done)
            if i > 0 and not np.isnan(fisher[i-1]):
                if fisher[i-1] < 0 and fisher[i] >= 0:
                    tp_triggered = True
        
        if in_position and position_side < 0:
            # Exit when Fisher crosses back below 0
            if i > 0 and not np.isnan(fisher[i-1]):
                if fisher[i-1] > 0 and fisher[i] <= 0:
                    tp_triggered = True
        
        if tp_triggered:
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
    
    return signals