#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, price > 12h EMA50, and volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low, price < 12h EMA50, and volume > 1.5x 20-period average
# Exit when price returns to Donchian midpoint (mean reversion) or trend reverses
# Position size: 0.25 to limit drawdown and reduce trade frequency

name = "4h_Donchian_EMA_Volume_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # 4h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # 4h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h data for EMA50 trend confirmation (higher timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # 12h EMA50 for stronger trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, above both EMAs, volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema50[i] and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, below both EMAs, volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema50[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend turns bearish
            if (close[i] < donchian_mid[i]) or (close[i] < ema50[i]) or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend turns bullish
            if (close[i] > donchian_mid[i]) or (close[i] > ema50[i]) or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals