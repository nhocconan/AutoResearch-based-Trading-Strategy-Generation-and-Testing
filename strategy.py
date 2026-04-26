#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with weekly trend filter (price vs weekly VWAP) and volume confirmation.
Long when price breaks above R1 in weekly bullish bias with volume spike.
Short when price breaks below S1 in weekly bearish bias with volume spike.
Weekly trend filter avoids counter-trend trades in bear markets. Volume spike confirms institutional interest.
Designed for lower trade frequency on 12h timeframe (target: 12-37 trades/year) to minimize fee drag.
Discrete position sizing (0.25) reduces churn. Works in bull/bear by following weekly bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data to build weekly data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly VWAP from daily data (7-day rolling window)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vol_1d = df_1d['volume'].values
    tp_vol_1d = typical_price_1d.values * vol_1d
    
    tp_vol_sum = pd.Series(tp_vol_1d).rolling(window=7, min_periods=7).sum().values
    vol_sum = pd.Series(vol_1d).rolling(window=7, min_periods=7).sum().values
    weekly_vwap = tp_vol_sum / vol_sum
    weekly_vwap = np.where(vol_sum == 0, np.nan, weekly_vwap)
    
    # Weekly trend: price above/below weekly VWAP
    weekly_bullish = df_1d['close'].values > weekly_vwap
    weekly_bearish = df_1d['close'].values < weekly_vwap
    
    # Align weekly trend to 12h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    # Get 1d data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_range * 1.1 / 12)
    camarilla_s1 = prev_close - (prev_range * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 2 for shift, 20 for volume MA)
    start_idx = max(2, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with weekly bullish bias and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                weekly_bullish_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with weekly bearish bias and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  weekly_bearish_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Camarilla S1 OR weekly bias turns bearish
            if (close[i] < camarilla_s1_aligned[i] or not weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Camarilla R1 OR weekly bias turns bullish
            if (close[i] > camarilla_r1_aligned[i] or not weekly_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0