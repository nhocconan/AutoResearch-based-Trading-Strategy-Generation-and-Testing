#!/usr/bin/env python3
"""
4d_donchian_breakout_1d_trend_volume_v2 (4h timeframe)
Hypothesis: Breakouts from Donchian(20) channel on 4h filtered by 1-day EMA50 trend and volume confirmation.
Long when price breaks above upper Donchian with volume > average and price above 1d EMA50.
Short when price breaks below lower Donchian with volume > average and price below 1d EMA50.
Designed for 20-30 trades/year on 4h with clear logic that works in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4d_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > high_max[i-1]
        bearish_breakout = close[i] < low_min[i-1]
        
        # 1d trend filter
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish breakout or trend turns bearish
            if bearish_breakout or below_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish breakout or trend turns bullish
            if bullish_breakout or above_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bullish Donchian breakout with volume confirmation and bullish trend
            if bullish_breakout and vol_confirmed and above_1d_ema50:
                position = 1
                signals[i] = 0.25
            # Short: bearish Donchian breakout with volume confirmation and bearish trend
            elif bearish_breakout and vol_confirmed and below_1d_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals