#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d Williams %R extreme filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on the daily timeframe.
# Breakout above R3 with Williams %R < -80 (oversold reversal) or below S3 with Williams %R > -20 (overbought reversal)
# provides counter-trend entries with momentum confirmation. Volume spike validates breakout strength.
# Designed for fewer, higher-quality trades (target: 20-40/year) to minimize fee drag in ranging markets.

name = "4h_Camarilla_R3_S3_Breakout_1dWilliamsR_Extreme_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Williams %R and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Williams %R needs min_periods=14
        return np.zeros(n)
    
    # 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d data for Camarilla pivot calculation (yesterday's OHLC)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 4
    s3 = close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need sufficient history for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extremes: < -80 oversold, > -20 overbought
        wr_oversold = williams_r_aligned[i] < -80
        wr_overbought = williams_r_aligned[i] > -20
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3, volume spike, Williams %R oversold (contrarian bounce)
            if close[i] > r3_aligned[i] and vol_spike and wr_oversold:
                signals[i] = 0.25
                position = 1
            # Short: break below S3, volume spike, Williams %R overbought (contrarian fade)
            elif close[i] < s3_aligned[i] and vol_spike and wr_overbought:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below R3 or Williams %R becomes overbought
            if close[i] < r3_aligned[i] or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above S3 or Williams %R becomes oversold
            if close[i] > s3_aligned[i] or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals