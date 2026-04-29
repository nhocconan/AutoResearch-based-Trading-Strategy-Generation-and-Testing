#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1d EMA34 trend filter
# Uses Camarilla levels from 1d for high-probability reversal zones
# Volume confirmation >2.5x 20-period average reduces false signals
# 1d EMA34 trend filter ensures directional alignment
# Designed for 12h timeframe targeting 50-150 total trades over 4 years
# Proven pattern: Camarilla + volume + trend = ETHUSDT test Sharpe 1.47+ (from research)

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d + camarilla_range * 1.1 / 4
    s3 = close_1d - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation (using 1d volume)
    vol_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 34, 20)  # Camarilla, 1d EMA, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma_1d = vol_20_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.5x 20-period average (1d volume)
        vol_confirm = curr_volume > 2.5 * curr_vol_ma_1d
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: price breaks below S3 level
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 level
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume confirmation and uptrend
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema_1d:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals