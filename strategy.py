#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA(50) trend filter and volume confirmation
# Long when price breaks above Camarilla R4 (stronger resistance) AND price > 1d EMA(50) AND volume > 1.8x 24-period average
# Short when price breaks below Camarilla S4 (stronger support) AND price < 1d EMA(50) AND volume > 1.8x 24-period average
# Uses tighter Camarilla levels (R4/S4) for fewer, higher-quality breakouts + volume confirmation to reduce false signals
# Discrete position sizing (0.25) minimizes fee drag. Works in both bull and bear by following HTF trend.
# Timeframe: 4h (primary), HTF: 1d for trend filter and Camarilla levels.

name = "4h_Camarilla_R4S4_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate Camarilla pivot levels from 1d data (R4/S4 = stronger levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla levels: R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    camarilla_r4 = close_1d_arr + (high_1d - low_1d) * 1.1
    camarilla_s4 = close_1d_arr - (high_1d - low_1d) * 1.1
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 1.8x 24-period average
    vol_ma_24 = np.zeros(n)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price breaks below Camarilla S4
            # 2. Price < 1d EMA(50)
            if curr_close < curr_s4 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price breaks above Camarilla R4
            # 2. Price > 1d EMA(50)
            if curr_close > curr_r4 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation
            vol_spike = volume[i] > 1.8 * vol_ma_24[i] if i >= 24 and vol_ma_24[i] > 0 else False
            
            # Long entry: price breaks above Camarilla R4 AND price > 1d EMA(50) AND volume spike
            if curr_close > curr_r4 and curr_close > curr_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S4 AND price < 1d EMA(50) AND volume spike
            elif curr_close < curr_s4 and curr_close < curr_ema and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals