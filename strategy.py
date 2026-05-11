#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily close for EMA34 trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Previous day's OHLC for Camarilla (R3/S3 levels)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    prev_close_1d[0] = df_1d['close'].values[0]
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 6
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 6
    
    # Align to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R3 with volume AND daily uptrend (close > EMA34)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                close_1d[-1] > ema_34_1d[-1] if len(close_1d) > 0 else False):  # Simplified trend check
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume AND daily downtrend (close < EMA34)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  close_1d[-1] < ema_34_1d[-1] if len(close_1d) > 0 else False):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to EMA34 or opposite level
            if position == 1:
                # Exit long: price returns to EMA34 or touches S3
                if (close[i] < ema_34_aligned[i]) or (close[i] < camarilla_s3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA34 or touches R3
                if (close[i] > ema_34_aligned[i]) or (close[i] > camarilla_r3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals