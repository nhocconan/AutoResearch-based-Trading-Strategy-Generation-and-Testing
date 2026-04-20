#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(10) breakout with 1-day trend filter + volume confirmation
# In bull market (price > 1-day EMA50): buy breakouts above upper band
# In bear market (price < 1-day EMA50): sell breakdowns below lower band
# Volume confirmation: require volume > 1.8x 20-period average
# This strategy avoids overtrading by using tight bands (10-period) and strict volume filter
# Target: 80-120 total trades over 4 years (20-30/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load 12h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 10-period Donchian channels
    upper = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lower = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend
        is_bull = close[i] > ema50_1d_aligned[i]
        is_bear = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: bull market + breakout above upper band + volume
            long_signal = False
            if has_volume and is_bull and price > upper[i]:
                long_signal = True
            
            # Enter short: bear market + breakdown below lower band + volume
            short_signal = False
            if has_volume and is_bear and price < lower[i]:
                short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below lower band (reversal signal)
            exit_signal = False
            if price < lower[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above upper band (reversal signal)
            exit_signal = False
            if price > upper[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian10_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0