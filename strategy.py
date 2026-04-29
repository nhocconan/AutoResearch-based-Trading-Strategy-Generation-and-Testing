#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla R3/S3 breakout continuation + volume confirmation
# Long when price breaks above 6h Donchian high AND price > 1d Camarilla R3 AND volume > 1.5x 20-period average
# Short when price breaks below 6h Donchian low AND price < 1d Camarilla S3 AND volume > 1.5x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear markets by
# combining breakout momentum with Camarilla pivot structure for institutional-level breakout confirmation.

name = "6h_Donchian20_Breakout_1dCamarillaR3S3_Volume_v1"
timeframe = "6h"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2.0)
    s3 = pivot - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 30, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # 6h Donchian(20) channels - calculate using lookback window
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = curr_high
            donchian_low = curr_low
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR Camarilla S3
            if curr_close < donchian_low or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR Camarilla R3
            if curr_close > donchian_high or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > Camarilla R3 AND volume spike
            if (curr_close > donchian_high and 
                curr_close > curr_r3 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < Camarilla S3 AND volume spike
            elif (curr_close < donchian_low and 
                  curr_close < curr_s3 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals