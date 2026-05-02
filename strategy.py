#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses 6h primary timeframe to reduce trade frequency vs lower TFs (target: 12-30 trades/year)
# Weekly pivot (from 1w) provides structural bias: long above weekly pivot, short below
# Donchian(20) breakout on 6f captures momentum in direction of weekly trend
# Volume confirmation (>2.0 * 20-period EMA) ensures institutional participation
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Weekly pivot filter avoids counter-trend trades in ranging markets
# Works in bull markets via breakout continuation and bear markets via trend alignment

name = "6h_Donchian20_WeeklyPivot_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot (based on previous weekly bar)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous weekly bar OHLC for pivot calculation
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_pivot = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        weekly_pivot[i] = (prev_high_1w[i] + prev_low_1w[i] + prev_close_1w[i]) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h timeframe
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            window_high = high[i-lookback+1:i+1]
            window_low = low[i-lookback+1:i+1]
            highest_high[i] = np.max(window_high)
            lowest_low[i] = np.min(window_low)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (6h * 5 = 30 periods = ~7.5 days)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias from weekly pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Donchian high with volume spike
                if close[i] > highest_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Donchian low with volume spike
                if close[i] < lowest_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around weekly pivot
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low or price below weekly pivot
            if close[i] < lowest_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or price above weekly pivot
            if close[i] > highest_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals