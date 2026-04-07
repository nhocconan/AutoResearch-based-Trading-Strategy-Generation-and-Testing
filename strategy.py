#!/usr/bin/env python3
"""
4h_river_2_bank_v1
Hypothesis: Combines 4h price action with 1d trend and volume confirmation. 
Long when price touches 4h 20-period EMA support in 1d uptrend with volume confirmation.
Short when price touches 4h 20-period EMA resistance in 1d downtrend with volume confirmation.
Uses the 'river to bank' concept where price moves from trend (river) to mean-reversion (bank).
Designed for low trade frequency and high win rate in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_river_2_bank_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for dynamic support/resistance
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price position relative to EMA20
        price_vs_ema = (close[i] - ema_20[i]) / ema_20[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below EMA20 or trend turns down
            if price_vs_ema < -0.005 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above EMA20 or trend turns up
            if price_vs_ema > 0.005 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price near EMA20 support in 1d uptrend with volume
            if (price_vs_ema > -0.01 and price_vs_ema < 0.01 and  # near EMA20
                close[i] > ema_50_1d_aligned[i] and  # 1d uptrend
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short: price near EMA20 resistance in 1d downtrend with volume
            elif (price_vs_ema > -0.01 and price_vs_ema < 0.01 and  # near EMA20
                  close[i] < ema_50_1d_aligned[i] and  # 1d downtrend
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals