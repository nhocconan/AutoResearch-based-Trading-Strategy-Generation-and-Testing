#!/usr/bin/env python3
"""
12h Bollinger Band Breakout with Volume Spike and Daily Trend Filter
Long: Close breaks above upper BB(20,2) + volume > 2x 12h volume MA(4) + price > 1d EMA50
Short: Close breaks below lower BB(20,2) + volume > 2x 12h volume MA(4) + price < 1d EMA50
Exit: Close crosses back inside the Bollinger Bands (mean reversion in chop)
Target: 15-25 trades/year per symbol (60-100 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) on 12h
    bb_period = 20
    bb_mult = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_bb = (sma + bb_mult * std).values
    lower_bb = (sma - bb_mult * std).values
    
    # 12h volume moving average (4-period)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, bb_period)  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: break above upper BB + volume spike + 1D uptrend
            if price > upper_bb[i] and vol > 2.0 * vol_ma_val and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower BB + volume spike + 1D downtrend
            elif price < lower_bb[i] and vol > 2.0 * vol_ma_val and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside BB (mean reversion)
            if price < upper_bb[i] and price > lower_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside BB (mean reversion)
            if price < upper_bb[i] and price > lower_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerBreakout_VolumeSpike_1DTrend"
timeframe = "12h"
leverage = 1.0