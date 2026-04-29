#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(50) trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND price > 1d EMA(50) AND volume > 2.0x 20-period average
# Short when price breaks below 1d Camarilla S3 AND price < 1d EMA(50) AND volume > 2.0x 20-period average
# Uses tighter R3/S3 levels (vs R1/S1) and higher volume threshold to reduce trade frequency.
# Proven pattern from DB: 4H_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Dyn achieved test Sharpe=1.901 on SOLUSDT

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels (based on previous 1d bar's OHLC)
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1 / 4
    r3_1d = close_1d + camarilla_range
    s3_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_r3 = r3_1d_aligned[i]
        curr_s3 = s3_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR price < 1d EMA(50)
            if curr_close < curr_s3 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR price > 1d EMA(50)
            if curr_close > curr_r3 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1d EMA(50) AND volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND price < 1d EMA(50) AND volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals