#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets
# 12h EMA50 filter ensures alignment with higher timeframe trend
# Volume > 1.5x 20-period average confirms breakout strength
# Works in both bull and bear markets by trading breakouts in direction of 12h trend
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h timeframe for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 20-period Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Upper band: highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band in uptrend with volume
            long_signal = False
            if has_volume and is_uptrend and price > donchian_high[i]:
                long_signal = True
            
            # Enter short: price breaks below lower Donchian band in downtrend with volume
            short_signal = False
            if has_volume and is_downtrend and price < donchian_low[i]:
                short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian band OR trend reversal
            exit_signal = False
            if price < donchian_low[i] or (not is_uptrend and has_volume):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian band OR trend reversal
            exit_signal = False
            if price > donchian_high[i] or (not is_downtrend and has_volume):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hTrendFilter_Volume"
timeframe = "4h"
leverage = 1.0