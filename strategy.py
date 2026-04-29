#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1h for entry timing but gets signal direction from 4h EMA50 to reduce trade frequency
# Long when: price breaks above Camarilla R3 AND price > 4h EMA50 AND volume > 1.5x 20-period average
# Short when: price breaks below Camarilla S3 AND price < 4h EMA50 AND volume > 1.5x 20-period average
# Exit when price returns to Camarilla Pivot level or opposite Camarilla level is touched
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in both bull/bear markets by combining breakout momentum with trend filter

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels using previous day's OHLC (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H-L = range
    camarilla_range = prev_high - prev_low
    camarilla_h3 = prev_close + camarilla_range * 1.1 / 4
    camarilla_l3 = prev_close - camarilla_range * 1.1 / 4
    camarilla_h4 = prev_close + camarilla_range * 1.1 / 2
    camarilla_l4 = prev_close - camarilla_range * 1.1 / 2
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_vol_ma = volume_ma_20[i]
        curr_h3 = camarilla_h3_aligned[i]
        curr_l3 = camarilla_l3_aligned[i]
        curr_h4 = camarilla_h4_aligned[i]
        curr_l4 = camarilla_l4_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > 1.5 * curr_vol_ma if curr_vol_ma > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions: price below pivot OR price touches L4 (strong reversal)
            if curr_close < curr_pivot or curr_low <= curr_l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: price above pivot OR price touches H4 (strong reversal)
            if curr_close > curr_pivot or curr_high >= curr_h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above H3 AND price > 4h EMA50 AND volume spike
            if (curr_high > curr_h3 and 
                curr_close > curr_ema_4h and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below L3 AND price < 4h EMA50 AND volume spike
            elif (curr_low < curr_l3 and 
                  curr_close < curr_ema_4h and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals