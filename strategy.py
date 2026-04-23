#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R4 AND close > 1d EMA34 AND volume > 1.5x 24-period average.
Short when price breaks below Camarilla S4 AND close < 1d EMA34 AND volume > 1.5x 24-period average.
Exit when price crosses 1d EMA34 in opposite direction.
Uses discrete position sizing (0.25) to balance return and drawdown. Targets 12-25 trades/year per symbol.
Camarilla R4/S4 levels represent stronger intraday support/resistance than R3/S3, reducing false breakouts.
1d EMA34 provides a smooth trend filter aligned with the 12h timeframe.
Volume confirmation at 1.5x ensures only significant breakouts are taken.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data for Camarilla levels and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous 1d bar)
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    cam_r4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.5  # R4 level
    cam_s4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.5  # S4 level
    cam_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2    # H3 for reference
    cam_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2    # L3 for reference
    
    # Align Camarilla levels to 12h timeframe
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    cam_h3_aligned = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_l3_aligned = align_htf_to_ltf(prices, df_1d, cam_l3)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (24-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(cam_r4_aligned[i]) or np.isnan(cam_s4_aligned[i]) or 
            np.isnan(cam_h3_aligned[i]) or np.isnan(cam_l3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R4 AND close > 1d EMA34 AND volume spike
            if (price > cam_r4_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S4 AND close < 1d EMA34 AND volume spike
            elif (price < cam_s4_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses 1d EMA34 in opposite direction
            if position == 1 and close[i] < ema34_1d_aligned[i]:
                exit_signal = True
            elif position == -1 and close[i] > ema34_1d_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R4S4_1dEMA34_VolumeSpike_SessionFilter"
timeframe = "12h"
leverage = 1.0