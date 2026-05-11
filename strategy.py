#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels (R3, S3)
    # R3 = Close + 1.1 * (High - Low) / 2
    # S3 = Close - 1.1 * (High - Low) / 2
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and previous day)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 with volume
            # AND price above daily EMA34 (uptrend filter)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 with volume
            # AND price below daily EMA34 (downtrend filter)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Camarilla S4/S5 (midpoint) or opposite level
            if position == 1:
                # Exit long: price returns to S4 (Close - 1.1*(H-L)/4) or breaks S3
                camarilla_s4 = df_1d['close'].shift(1).values - 1.1 * (prev_high - prev_low) / 4
                camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
                if (not np.isnan(camarilla_s4_aligned[i]) and 
                    (close[i] < camarilla_s4_aligned[i] or close[i] < camarilla_s3_aligned[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R4 (Close + 1.1*(H-L)/4) or breaks R3
                camarilla_r4 = df_1d['close'].shift(1).values + 1.1 * (prev_high - prev_low) / 4
                camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
                if (not np.isnan(camarilla_r4_aligned[i]) and 
                    (close[i] > camarilla_r4_aligned[i] or close[i] > camarilla_r3_aligned[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals