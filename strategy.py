#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and 1d volume spike
# Uses 1d Camarilla levels for high-probability reversal zones, 4h EMA200 for trend filter
# Volume confirmation (>1.8x 24-period average) on 1h timeframe to avoid false breakouts
# Session filter (08-20 UTC) to reduce noise. Designed for 1h timeframe targeting 60-150 trades over 4 years (15-37/year)
# Works in both bull and bear: trend filter ensures we only trade with higher timeframe momentum

name = "1h_Camarilla_R3S3_Breakout_4hEMA200_1dVolSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1d) < 1 or len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d + camarilla_range * 1.1 / 4
    s3 = close_1d - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 1d average volume for spike detection (24-period = 24*1h bars in 1d)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 50, 200, 24)  # Camarilla, 4h EMA, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session or any required data is NaN
        if not in_session[i] or (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
                                 np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_4h = ema_200_4h_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_24[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
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
            # Long entry: price breaks above R3 with volume confirmation and uptrend (close > 4h EMA200)
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema_4h:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and downtrend (close < 4h EMA200)
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema_4h:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals