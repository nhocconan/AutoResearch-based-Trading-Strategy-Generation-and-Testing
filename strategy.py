#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v2
Hypothesis: Weekly trend filter + daily Donchian breakout with volume confirmation.
Long when price breaks above daily Donchian(20) with volume > average and price above weekly EMA50.
Short when price breaks below daily Donchian(20) with volume > average and price below weekly EMA50.
Designed for 15-25 trades/year on 1d with clear logic that works in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v2"
timeframe = "1d"
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
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Donchian breakout conditions (using previous bar's levels)
        bullish_breakout = close[i] > high_max[i-1]
        bearish_breakout = close[i] < low_min[i-1]
        
        # Weekly trend filter
        above_1w_ema50 = close[i] > ema50_1w_aligned[i]
        below_1w_ema50 = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish breakout or trend turns bearish
            if bearish_breakout or below_1w_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish breakout or trend turns bullish
            if bullish_breakout or above_1w_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bullish Donchian breakout with volume confirmation and bullish trend
            if bullish_breakout and vol_confirmed and above_1w_ema50:
                position = 1
                signals[i] = 0.25
            # Short: bearish Donchian breakout with volume confirmation and bearish trend
            elif bearish_breakout and vol_confirmed and below_1w_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals