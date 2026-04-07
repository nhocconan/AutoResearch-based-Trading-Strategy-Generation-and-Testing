#!/usr/bin/env python3
"""
1d_donchian_1w_trend_volume_v1
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter (EMA50) and volume confirmation.
In trending markets (price > weekly EMA50), buy breakouts above upper band; in ranging markets,
fade at lower band with volume confirmation. Works in both bull and bear by adapting to weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly EMA50 (trend change)
            if close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above weekly EMA50 (trend change)
            if close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # In uptrend (price > weekly EMA50): buy breakout above upper band
            if (close[i] > high_max[i] and 
                vol_confirm and 
                close[i] > ema50_1d[i]):
                position = 1
                signals[i] = 0.25
            # In downtrend (price < weekly EMA50): sell breakout below lower band
            elif (close[i] < low_min[i] and 
                  vol_confirm and 
                  close[i] < ema50_1d[i]):
                position = -1
                signals[i] = -0.25
            # In ranging market (near weekly EMA50): fade at Donchian bands with volume
            elif (abs(close[i] - ema50_1d[i]) < 0.02 * ema50_1d[i]):  # Within 2% of weekly EMA
                # Fade lower band (long) with volume confirmation
                if (close[i] <= low_min[i] and 
                    vol_confirm and 
                    close[i] > ema50_1d[i]):
                    position = 1
                    signals[i] = 0.25
                # Fade upper band (short) with volume confirmation
                elif (close[i] >= high_max[i] and 
                      vol_confirm and 
                      close[i] < ema50_1d[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals