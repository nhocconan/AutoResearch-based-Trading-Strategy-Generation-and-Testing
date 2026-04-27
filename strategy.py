#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike
Hypothesis: Donchian(20) breakouts on 6h aligned with weekly pivot trend (price vs weekly Camarilla pivot) and volume spikes capture strong momentum moves. Weekly pivot acts as dynamic support/resistance. Volume confirmation avoids false breakouts. Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend filter (Camarilla pivot)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    range_1w = high_1w - low_1w
    # Weekly Camarilla pivot levels (R3, S3) from prior week
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r3 = weekly_pivot + 1.125 * range_1w
    weekly_s3 = weekly_pivot - 1.125 * range_1w
    
    # Donchian(20) channels on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align weekly indicators to 6h timeframe
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)  # volume is LTF, but confirm using weekly avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian (20), weekly pivot (1), volume avg (20)
    start_idx = max(20, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        highest = highest_20[i]
        lowest = lowest_20[i]
        weekly_r3_val = weekly_r3_aligned[i]
        weekly_s3_val = weekly_s3_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs weekly Camarilla R3/S3
            bullish_bias = close_val > weekly_r3_val  # Above weekly R3 = bullish bias
            bearish_bias = close_val < weekly_s3_val  # Below weekly S3 = bearish bias
            
            if bullish_bias and vol_conf:
                # Long when price breaks above Donchian high with volume and bullish bias
                if close_val > highest:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif bearish_bias and vol_conf:
                # Short when price breaks below Donchian low with volume and bearish bias
                if close_val < lowest:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: stoploss (2.0*ATR) or Donchian low touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < lowest:  # Donchian low touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: stoploss (2.0*ATR) or Donchian high touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > highest:  # Donchian high touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0