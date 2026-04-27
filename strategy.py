#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_Volume
Hypothesis: 6h strategy using weekly Camarilla S4/R4 levels for breakout direction and 6h Donchian(20) for precise entry timing. 
Weekly pivot determines bias: long only when price above weekly S4, short only when below weekly R4. 
Entry on Donchian breakout in direction of weekly bias with volume confirmation (>1.5x 20-bar average). 
Exit on opposite Donchian breakout or weekly bias reversal. 
Designed for very low trade frequency (<15/year) to minimize fee drag in bear markets.
Works in bull by capturing breakouts above weekly resistance and in bear by shorting breakdowns below weekly support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla S4/R4 levels (weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly OHLC for Camarilla levels
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Weekly Camarilla levels: R4/S4 from weekly OHLC (strong breakout/breakdown levels)
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_r4 = c_1w + (h_1w - l_1w) * 1.1 / 2
    camarilla_s4 = c_1w - (h_1w - l_1w) * 1.1 / 2
    
    # 6h Donchian(20) for breakout entries
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need weekly data (no lookback) + Donchian(20) + volume avg (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r4_val = camarilla_r4[i]
        s4_val = camarilla_s4[i]
        upper_donch = donchian_high[i]
        lower_donch = donchian_low[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of weekly bias with volume confirmation
            # Long: price breaks above Donchian high AND above weekly S4 (bullish bias)
            long_condition = (close_val > upper_donch) and (close_val > s4_val) and vol_conf
            # Short: price breaks below Donchian low AND below weekly R4 (bearish bias)
            short_condition = (close_val < lower_donch) and (close_val < r4_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly bias turns bearish (price below R4)
            if (close_val < lower_donch) or (close_val < r4_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly bias turns bullish (price above S4)
            if (close_val > upper_donch) or (close_val > s4_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0