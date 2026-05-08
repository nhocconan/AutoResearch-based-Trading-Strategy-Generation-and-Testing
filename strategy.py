#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with daily price action
# We go long when price is above daily EMA(50) and market is trending (CHOP < 38.2)
# We go short when price is below daily EMA(50) and market is trending (CHOP < 38.2)
# We stay flat when market is ranging (CHOP > 61.8)
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Choppiness Index identifies market regime to avoid whipsaws in ranging markets.
# Daily EMA(50) provides trend direction filter.

name = "12h_Choppiness_Regime_DailyEMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data once for EMA and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Choppiness Index (14-period) on daily data
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_vals = df_1d['close'].values
    
    # Calculate True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close_vals, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close_vals, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR (14-period)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate Choppiness Index
    chop_period = 14
    chop = np.full_like(daily_close_vals, np.nan)
    
    for i in range(chop_period-1, len(daily_close_vals)):
        # Sum of ATR over the period
        sum_atr = np.sum(atr[i-chop_period+1:i+1])
        # Max high and min low over the period
        max_high = np.max(daily_high[i-chop_period+1:i+1])
        min_low = np.min(daily_low[i-chop_period+1:i+1])
        # Avoid division by zero
        if max_high - min_low > 0:
            chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_period)
    
    # Align indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Enter long: price above EMA50 and trending market (CHOP < 38.2)
            if close[i] > ema50_val and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA50 and trending market (CHOP < 38.2)
            elif close[i] < ema50_val and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below EMA50 OR market becomes ranging (CHOP > 61.8)
            if close[i] < ema50_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above EMA50 OR market becomes ranging (CHOP > 61.8)
            if close[i] > ema50_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals