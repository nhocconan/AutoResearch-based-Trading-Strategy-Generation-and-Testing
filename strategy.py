#!/usr/bin/env python3
# 1h_PV_Camarilla_T4_S4_Breakout_4hTrend_1dVolume
# Strategy: Trade PV_Camarilla pivot breakouts with 4h trend filter and 1d volume confirmation
# Long when price breaks above PV_Camarilla T4 level with 4h uptrend and above-average volume
# Short when price breaks below PV_Camarilla S4 level with 4h downtrend and above-average volume
# Exit when price crosses back through the PV_Camarilla PP (pivot point) level
# Uses volume surge to confirm breakouts and 4h trend to avoid counter-trend trades
# Designed for 1h timeframe with selective entries to meet trade frequency targets

name = "1h_PV_Camarilla_T4_S4_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate PV_Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    t4 = pp + range_val * 1.1 / 2.0  # T4 level
    s4 = pp - range_val * 1.1 / 2.0  # S4 level
    
    # Align PV_Camarilla levels to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    t4_aligned = align_htf_to_ltf(prices, df_1d, t4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(t4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or
            np.isnan(prices['volume'].iloc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        current_volume = prices['volume'].iloc[i]
        
        if position == 0:
            # Enter long: price breaks above T4 with 4h uptrend and volume surge
            if (prices['high'].iloc[i] > t4_aligned[i] and 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[max(0, i-1)] and  # 4h EMA rising
                current_volume > vol_avg_20_1d_aligned[i] * 1.5):  # Volume 1.5x average
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S4 with 4h downtrend and volume surge
            elif (prices['low'].iloc[i] < s4_aligned[i] and 
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[max(0, i-1)] and  # 4h EMA falling
                  current_volume > vol_avg_20_1d_aligned[i] * 1.5):  # Volume 1.5x average
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back through PP level
            if prices['low'].iloc[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses back through PP level
            if prices['high'].iloc[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals