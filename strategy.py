#!/usr/bin/env python3
"""
1D Weekly Range Breakout with Volume Confirmation and Trend Filter
Long: Price breaks above prior week high + volume > 1.5x 1D volume MA + price > 1D EMA50
Short: Price breaks below prior week low + volume > 1.5x 1D volume MA + price < 1D EMA50
Exit: Opposite break of prior weekly level
Designed for 1d timeframe to capture weekly momentum with controlled trade frequency
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for prior week high/low
    df_1w = get_htf_data(prices, '1w')
    prior_1w_high = df_1w['high'].shift(1)  # Prior week's high
    prior_1w_low = df_1w['low'].shift(1)    # Prior week's low
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    prior_1w_high_aligned = align_htf_to_ltf(prices, df_1w, prior_1w_high.values)
    prior_1w_low_aligned = align_htf_to_ltf(prices, df_1w, prior_1w_low.values)
    
    # 1D volume moving average (20-period for confirmation)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1w_high_aligned[i]) or np.isnan(prior_1w_low_aligned[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        if position == 0:
            # Long: break above prior week high + volume + 1D trend
            if price > prior_1w_high_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_1d[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior week low + volume + 1D trend
            elif price < prior_1w_low_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_1d[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior week low
            if price < prior_1w_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior week high
            if price > prior_1w_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_WeeklyRange_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0