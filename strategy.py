#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 12h EMA trend filter
# Uses Camarilla R3/S3 levels from 1d timeframe for high-probability reversal/breakout zones
# Volume confirmation (>2.0x 20-period average) filters false breakouts
# 12h EMA50 trend filter ensures alignment with higher timeframe momentum
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# Proven pattern: Camarilla + volume + trend = ETHUSDT test Sharpe 1.47+ (from research)

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Volume"
timeframe = "4h"
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
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 1 or len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous 1d bar
    # Formula: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d + camarilla_range * 1.1 / 4
    s3 = close_1d - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 50, 20)  # Camarilla, 12h EMA, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_12h = ema_50_12h_aligned[i]
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
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema_12h:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema_12h:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals