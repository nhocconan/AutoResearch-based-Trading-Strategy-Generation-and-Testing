#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Enter long when price breaks above Donchian(20) high with weekly pivot above prior week close and volume > 2x 20-bar average.
# Enter short when price breaks below Donchian(20) low with weekly pivot below prior week close and volume > 2x 20-bar average.
# Exit when price retraces to the Donchian(20) midpoint.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Weekly pivot provides higher timeframe bias; Donchian breakout captures momentum; volume spike filters weak breakouts.
# Works in bull markets (strong breakouts with upward bias) and bear markets (strong breakdowns with downward bias).

name = "6h_Donchian20_WeeklyCPR_Pivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly CPR (Central Pivot Range) - using weekly high/low/close
    # Resample to weekly using actual Binance weekly data via HTF
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close for CPR
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Align weekly data to 6h
    prev_week_high_aligned = align_htf_to_ltf(prices, df_1w, prev_week_high)
    prev_week_low_aligned = align_htf_to_ltf(prices, df_1w, prev_week_low)
    prev_week_close_aligned = align_htf_to_ltf(prices, df_1w, prev_week_close)
    
    # Weekly CPR: pivot = (H+L+C)/3, BC = (H+L)/2, TC = (C - BC) + pivot
    weekly_pivot = (prev_week_high_aligned + prev_week_low_aligned + prev_week_close_aligned) / 3
    weekly_bc = (prev_week_high_aligned + prev_week_low_aligned) / 2  # Bottom of CPR
    weekly_tc = (prev_week_close_aligned - weekly_bc) + weekly_pivot   # Top of CPR
    
    # Weekly bias: above TC = bullish, below BC = bearish, inside = neutral
    weekly_bullish = weekly_pivot > prev_week_close_aligned  # Simplified: pivot above prior close
    weekly_bearish = weekly_pivot < prev_week_close_aligned  # Pivot below prior close
    
    # Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(weekly_pivot[i]) or
            np.isnan(weekly_bc[i]) or np.isnan(weekly_tc[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, weekly bullish, volume confirm
            if price > highest_high[i] and weekly_bullish[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, weekly bearish, volume confirm
            elif price < lowest_low[i] and weekly_bearish[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals