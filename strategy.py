#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla pivots identify key support/resistance levels where price often reverses or accelerates
# R3/S3 are the outer bands - breakouts suggest strong momentum continuation
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
# Volume confirmation (>1.8x 20-period average) filters weak breakouts
# Designed for low-frequency, high-conviction trades on daily timeframe
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Need at least 1 day of prior data to calculate Camarilla levels
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla pivot levels from previous day
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla formula
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # R3 and S3 levels (outer bands)
        r3 = prev_close + range_hl * 1.1 / 2.0
        s3 = prev_close - range_hl * 1.1 / 2.0
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Calculate 20-period average volume for confirmation
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.mean(volume[:i]) if i > 0 else volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = curr_volume > 1.8 * vol_ma_20
        
        # Skip if EMA data not ready
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_ema_1w = ema_50_1w_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below R3 (failed breakout) OR closes below weekly EMA50
            if curr_close < r3 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 (failed breakout) OR closes above weekly EMA50
            if curr_close > s3 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 + above weekly EMA50 + volume confirmation
            if (curr_close > r3 and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + below weekly EMA50 + volume confirmation
            elif (curr_close < s3 and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals