#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate previous day's OHLC for Camarilla levels
    # Use previous day's data to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for current day using previous day's data
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    high_low_range = prev_high - prev_low
    r3 = prev_close + high_low_range * 1.1 / 2
    s3 = prev_close - high_low_range * 1.1 / 2
    r4 = prev_close + high_low_range * 1.1
    s4 = prev_close - high_low_range * 1.1
    
    # Align to daily timeframe (no shift needed as we used previous day's data)
    # The levels are already for the current day based on previous day's OHLC
    r3_aligned = r3
    s3_aligned = s3
    r4_aligned = r4
    s4_aligned = s4
    
    # Get weekly trend filter (1w EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ratio[i])):
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
            # Long: Price breaks above R3 with volume, weekly trend up
            if (close[i] > r3_aligned[i] and 
                volume_surge and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume, weekly trend down
            elif (close[i] < s3_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches R4/S4 or reverses against weekly trend
            if position == 1:
                # Exit long: price reaches R4 or closes below weekly EMA
                if (close[i] >= r4_aligned[i]) or (close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches S4 or closes above weekly EMA
                if (close[i] <= s4_aligned[i]) or (close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals