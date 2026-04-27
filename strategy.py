#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Uses 4h Camarilla pivot levels (R1/S1) for breakout entries on 1h timeframe with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above R1 AND 4h close > EMA34 (uptrend) AND volume > 2.0 * 20-period average.
Short when price breaks below S1 AND 4h close < EMA34 (downtrend) AND volume > 2.0 * 20-period average.
Exit when price returns to the pivot level (R1 for longs, S1 for shorts) OR trend reverses.
Session filter: 08-20 UTC to reduce noise. Position size: 0.20.
Designed for 1h timeframe to achieve 60-150 total trades over 4 years with low fee drag.
Uses 4h for signal direction (trend + Camarilla levels), 1h only for entry timing.
Works in both bull and bear markets by following 4h trend while using Camarilla levels for precise breakout entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA34 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla pivot levels: R1, S1
    # Camarilla formulas: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) * 1.1 / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need 4h EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirm[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 with 4h trend filter AND volume
            # Long: price breaks above R1 (minor resistance) AND 4h uptrend AND volume
            long_condition = (close_val > r1_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S1 (minor support) AND 4h downtrend AND volume
            short_condition = (close_val < s1_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to R1 level OR trend breaks
            exit_condition = (close_val <= r1_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to S1 level OR trend breaks
            exit_condition = (close_val >= s1_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0