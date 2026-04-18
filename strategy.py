#!/usr/bin/env python3
"""
4h Donchian Breakout + Daily EMA Trend + Volume Spike
Hypothesis: Donchian(20) breakouts on 4h capture trend continuations. Daily EMA(50) filters for higher timeframe trend direction, and volume spikes confirm institutional participation. This combination works in both bull and bear markets by requiring alignment with daily trend and volume confirmation, reducing false breakouts. Designed for 20-50 trades/year on 4h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend filter
    ema_50_d = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_d, ema_50_d)
    
    # Donchian channels on 4h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # enough for Donchian and EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        ema_trend = ema_50_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above daily EMA (uptrend)
            if price > upper and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below daily EMA (downtrend)
            elif price < lower and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to lower Donchian or breaks below daily EMA
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to upper Donchian or breaks above daily EMA
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_DailyEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0