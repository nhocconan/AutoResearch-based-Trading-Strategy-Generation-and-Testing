#!/usr/bin/env python3
"""
12h_1d_CombinedMomentum_Signal_V1
Hypothesis: Combine 12h price momentum (ROC12) with 1d trend filter (EMA50) and volume confirmation to capture medium-term trends. Designed for low trade frequency (target: 12-37/year) in 12h timeframe. Works in bull markets via momentum continuation and in bear markets via trend-following with strict filters to avoid whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Calculate daily EMA50 for trend filter
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periodas=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h ROC12 for momentum
    roc12 = np.zeros_like(close)
    for i in range(12, n):
        roc12[i] = (close[i] - close[i-12]) / close[i-12] * 100
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema50_daily_aligned[i]) or np.isnan(roc12[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_daily_aligned[i]
        mom = roc12[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: positive momentum + above daily EMA50 + volume
            if mom > 0.5 and price > ema50 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: negative momentum + below daily EMA50 + volume
            elif mom < -0.5 and price < ema50 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: momentum turns negative or price drops below EMA50
            if mom < -0.2 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum turns positive or price rises above EMA50
            if mom > 0.2 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_CombinedMomentum_Signal_V1"
timeframe = "12h"
leverage = 1.0