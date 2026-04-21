#!/usr/bin/env python3
"""
1d_1w_DonchianBreakout_VolumeTrend_v1
Hypothesis: Use weekly Donchian channel breakout with volume confirmation and trend filter.
Long when price breaks above weekly upper Donchian channel (20-period) with volume > 1.5x 20-day average and daily close > weekly EMA20.
Short when price breaks below weekly lower Donchian channel with volume > 1.5x 20-day average and daily close < weekly EMA20.
Exit when price crosses weekly midline (average of upper and lower bands).
Designed for fewer trades (~10-25/year) to reduce fee drag and work in both bull and bear markets by following higher timeframe trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Donchian channels and EMA
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper_donchian = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    middle_donchian = (upper_donchian + lower_donchian) / 2.0
    
    # Calculate weekly EMA20 for trend filter
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to daily timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_weekly, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_weekly, lower_donchian)
    middle_donchian_aligned = align_htf_to_ltf(prices, df_weekly, middle_donchian)
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(middle_donchian_aligned[i]) or np.isnan(ema20_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Price relative to weekly EMA for trend filter
        price_above_ema = price > ema20_weekly_aligned[i]
        price_below_ema = price < ema20_weekly_aligned[i]
        
        if position == 0:
            # Long conditions: break above upper Donchian + volume + price above weekly EMA
            if price > upper_donchian_aligned[i] and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower Donchian + volume + price below weekly EMA
            elif price < lower_donchian_aligned[i] and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly midline
            if price < middle_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly midline
            if price > middle_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_DonchianBreakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0