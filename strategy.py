#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from prior 12h period: long on break above R3 in uptrend, short on break below S3 in downtrend
# Volume confirmation (>1.3x 20-period average) ensures institutional participation
# Designed for 6h timeframe to capture medium-term swings with controlled trade frequency (~20-40 trades/year)
# Works in both bull and bear markets by aligning with 12h trend (EMA50) to avoid counter-trend trades

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from prior 12h bar
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_range = (high_12h - low_12h) * 1.1
    r3 = close_12h + camarilla_range * 1.1 / 4
    s3 = close_12h - camarilla_range * 1.1 / 4
    
    # AlCamarilla levels to 6h timeframe (delayed by one 12h bar for look-ahead avoidance)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: reverse signal on opposite Camarilla level break or trend change
        if position == 1:  # Long position
            # Exit: price breaks below S3 or trend turns down (price < EMA50)
            if curr_low < curr_s3 or curr_close < curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or trend turns up (price > EMA50)
            if curr_high > curr_r3 or curr_close > curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_confirm = curr_volume > 1.3 * curr_vol_ma
            
            # Long entry: price breaks above R3 in uptrend (price > EMA50)
            if vol_confirm and curr_close > curr_ema50_12h:
                if curr_high > curr_r3:  # Break above R3 level
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below S3 in downtrend (price < EMA50)
            elif vol_confirm and curr_close < curr_ema50_12h:
                if curr_low < curr_s3:  # Break below S3 level
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals