#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 trend filter and volume confirmation (>1.6x 20-period average)
# Camarilla pivot levels (R3/S3) act as strong support/resistance; breakouts indicate institutional interest
# 12h EMA50 filters for trend alignment to avoid counter-trend trades; volume confirmation ensures follow-through
# Discrete sizing (0.25) minimizes fee churn; AT
# Works in both bull/bear markets: breakouts capture volatility expansion, regime filters adapt to conditions
# Target: 80-180 total trades over 4 years (20-45/year) on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
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
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Daily OHLC for Camarilla levels (using 1d data from HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # We'll use the previous day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align to 4h timeframe (using previous day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 12h EMA50, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.6x 20-period average
        vol_confirm = curr_volume > 1.6 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below S3 OR loss of volume confirmation
            if curr_close < curr_s3 or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 OR loss of volume confirmation
            if curr_close > curr_r3 or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: breakout above R3 + above 12h EMA50 + volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_12h and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below S3 + below 12h EMA50 + volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_12h and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals