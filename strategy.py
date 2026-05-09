#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # 6h Camarilla pivot levels from previous 6h bar
    # Pivot calculation: (H + L + C) / 3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    r4 = pivot + range_hl * 1.1 / 2
    s4 = pivot - range_hl * 1.1 / 2
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ema20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # enough for EMA34 and roll
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: break above R3 with volume, in 1d uptrend
            if (price > r3[i] and vol_filter[i] and price > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: break below S3 with volume, in 1d downtrend
            elif (price < s3[i] and vol_filter[i] and price < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price falls back to pivot or 1d trend fails
            if (price < pivot[i] or price < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back to pivot or 1d trend fails
            if (price > pivot[i] or price > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals