#!/usr/bin/env python3
"""
12h 1W High/Low Breakout with Volume and 1D Trend Filter
Long: Price breaks above prior 1W high + volume > 1.5x 12h volume MA + price > 1D EMA50
Short: Price breaks below prior 1W low + volume > 1.5x 12h volume MA + price < 1D EMA50
Exit: Opposite break of prior 1W level
Uses 1D EMA50 for trend filter and 1W for structure - reduces false breakouts in chop
Target: 15-25 trades/year per symbol
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
    
    # Get 1W data for prior high/low (structure)
    df_1w = get_htf_data(prices, '1w')
    prior_1w_high = df_1w['high'].shift(1)  # Prior week's high
    prior_1w_low = df_1w['low'].shift(1)    # Prior week's low
    
    prior_1w_high_aligned = align_htf_to_ltf(prices, df_1w, prior_1w_high.values)
    prior_1w_low_aligned = align_htf_to_ltf(prices, df_1w, prior_1w_low.values)
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume moving average (24-period for confirmation)
    df_12h = get_htf_data(prices, '12h')
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean()
    volume_ma_24_12h = align_htf_to_ltf(prices, df_12h, volume_ma_24.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1w_high_aligned[i]) or np.isnan(prior_1w_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_12h[i]
        
        if position == 0:
            # Long: break above prior 1W high + volume + 1D trend
            if price > prior_1w_high_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1W low + volume + 1D trend
            elif price < prior_1w_low_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior 1W low
            if price < prior_1w_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior 1W high
            if price > prior_1w_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Prior1W_HL_Breakout_Volume_1DTrend"
timeframe = "12h"
leverage = 1.0