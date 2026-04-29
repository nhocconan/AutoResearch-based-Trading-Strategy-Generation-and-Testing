#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above 12h Camarilla R3, 1d volume > 2.0x average, chop > 61.8 (range)
# Short when price breaks below 12h Camarilla S3, 1d volume > 2.0x average, chop > 61.8 (range)
# Exit when price reaches opposite Camarilla level (R3->S3 or S3->R3)
# Uses discrete position sizing (0.25) targeting 12-30 trades/year on 12h timeframe.
# Works in ranging markets by fading extremes with volume confirmation and regime filter.

name = "12h_Camarilla_R3S3_VolumeSpike_Chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align 1d Camarilla levels to 12h timeframe (no additional delay for pivots)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for price reference
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Chop regime filter: 14-period chop > 61.8 indicates ranging market
    TR1 = pd.Series(df_12h['high']).rolling(2).max() - pd.Series(df_12h['low']).rolling(2).min()
    TR2 = abs(pd.Series(df_12h['high']) - pd.Series(df_12h['close']).shift(1))
    TR3 = abs(pd.Series(df_12h['low']) - pd.Series(df_12h['close']).shift(1))
    TR = pd.concat([TR1, TR2, TR3], axis=1).max(axis=1)
    atr = TR.rolling(14, min_periods=14).mean()
    chop = 100 * np.log10(atr.rolling(14, min_periods=14).sum() / np.log10(14)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_values)
    
    # Calculate 20-period average volume for confirmation (1d)
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Volume and chop warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_chop = chop_aligned[i]
        curr_vol_ma = vol_ma_20_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reaches S3 level (opposite Camarilla level)
            if curr_low <= curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 level (opposite Camarilla level)
            if curr_high >= curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (strong filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            # Chop filter: chop > 61.8 indicates ranging market (good for mean reversion)
            chop_filter = curr_chop > 61.8
            
            # Long when price breaks above R3, volume confirmed, chop > 61.8 (range)
            if curr_high > curr_r3 and vol_confirmed and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, volume confirmed, chop > 61.8 (range)
            elif curr_low < curr_s3 and vol_confirmed and chop_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals