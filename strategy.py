#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. 
Breakouts above R3 or below S3 with 1d EMA trend alignment and volume spikes 
capture strong moves. Works in bull markets (long on upside breaks) and bear 
markets (short on downside breaks). Uses 4h timeframe targeting 75-200 trades 
over 4 years (19-50/year). Volume confirmation reduces false breakouts.
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
    
    # Get 1d data for Camarilla pivots and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily typical price for Camarilla
    typical_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_1d_vals = typical_1d.values
    
    # Calculate Camarilla levels: R3, S3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        camarilla_r3[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 4
        camarilla_s3[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 4
    
    # Align Camarilla levels to 4h (no extra delay needed for pivot points)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 34-period EMA on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike (on 4h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla R3, above 1d EMA, volume confirmation
            long_entry = (curr_close > camarilla_r3_val and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below Camarilla S3, below 1d EMA, volume confirmation
            short_entry = (curr_close < camarilla_s3_val and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla S3 OR below 1d EMA
            if curr_close < camarilla_s3_val or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla R3 OR above 1d EMA
            if curr_close > camarilla_r3_val or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0