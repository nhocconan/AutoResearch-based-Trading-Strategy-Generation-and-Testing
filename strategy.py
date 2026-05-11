#!/usr/bin/env python3
name = "12h_1D_Camarilla_R3S3_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R3 = close + (high - low) * 1.1/2
    # R2 = close + (high - low) * 1.1/4
    # R1 = close + (high - low) * 1.1/6
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.1/6
    # S2 = close - (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/2
    
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    R2 = prev_close + (prev_high - prev_low) * 1.1 / 4
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    PP = (prev_high + prev_low + prev_close) / 3
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    S2 = prev_close - (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align all Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 24-period average on 12h (24 * 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i]) or
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
            # Long: Price breaks above R3 with volume and above EMA34 trend
            if (close[i] > R3_aligned[i] and 
                volume_surge and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume and below EMA34 trend
            elif (close[i] < S3_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite S1/R1 level (mean reversion to pivot area)
            if position == 1:
                # Exit long: price touches or goes below S1
                if close[i] <= S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches or goes above R1
                if close[i] >= R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals