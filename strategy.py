#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h Camarilla levels for structure, 4h EMA50 for trend direction, 1h volume spike for entry timing
# Session filter (08-20 UTC) reduces noise. Designed for 1h timeframe targeting 60-150 trades over 4 years (15-37/year)
# Works in bull/bear: breaksout with volume in trending markets, avoids false breakouts in ranging markets

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
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
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3) from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels
    camarilla_range = (high_4h - low_4h) * 1.1
    r3 = close_4h + camarilla_range * 1.1 / 4
    s3 = close_4h - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period average volume for confirmation (1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 50, 20)  # Camarilla, 4h EMA, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: price breaks below S3 level
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 level
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume confirmation and uptrend
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema_4h:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema_4h:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals