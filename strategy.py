#!/usr/bin/env python3
"""
12h Camarilla R1S1 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Camarilla R1/S1 levels on 1w act as significant support/resistance. 
Break above R1 with volume and 1w EMA>50 (bullish trend) signals bullish momentum.
Break below S1 with volume and 1w EMA<50 signals bearish momentum.
Uses 12h timeframe for lower trade frequency. Works in bull/bear via EMA trend filter.
Volume spike confirms institutional participation. Target: 12-37 trades/year.
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
    
    # Get 1w data for Camarilla pivot calculation and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA and pivot
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot levels (based on previous week's OHLC)
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate pivot levels using previous week's data
    range_hl = prev_high - prev_low
    camarilla_r1 = prev_close + (range_hl * 1.1 / 2)  # R1 level
    camarilla_s1 = prev_close - (range_hl * 1.1 / 2)  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_value = ema_50_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter: price above/below EMA50
        bullish_trend = close[i] > ema_value
        bearish_trend = close[i] < ema_value
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND bullish trend
            long_condition = (curr_close > r1_level) and volume_spike and bullish_trend
            # Short: price breaks below S1 AND volume spike AND bearish trend
            short_condition = (curr_close < s1_level) and volume_spike and bearish_trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below S1 or trend turns bearish
            if curr_close <= s1_level or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above R1 or trend turns bullish
            if curr_close >= r1_level or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0