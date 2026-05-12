#/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA FOR CAMARILLA PIVOTS AND TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's Camarilla pivot levels
    # Using previous day's high, low, close
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla calculations
    range_prev = prev_high - prev_low
    camarilla_r3 = prev_close + (range_prev * 1.1 / 2)
    camarilla_s3 = prev_close - (range_prev * 1.1 / 2)
    camarilla_r4 = prev_close + (range_prev * 1.1)
    camarilla_s4 = prev_close - (range_prev * 1.1)
    
    # Align camarilla levels to 6h timeframe
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(r4_1d[i]) or 
            np.isnan(s4_1d[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume, trend up (price > EMA34)
            if (close[i] > r3_1d[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume, trend down (price < EMA34)
            elif (close[i] < s3_1d[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below S3 (reversal) or trend changes
            if (close[i] < s3_1d[i]) or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R3 (reversal) or trend changes
            if (close[i] > r3_1d[i]) or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals