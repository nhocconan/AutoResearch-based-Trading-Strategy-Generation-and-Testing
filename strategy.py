#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high, above 12h EMA50, and volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian low, below 12h EMA50, and volume > 1.5x 20-period average
# Exit when price crosses back below/above Donchian median OR EMA direction contradicts position
# Position size: 0.25 to manage drawdown
# Designed to work in trending markets via EMA filter and avoid false breakouts via volume confirmation

name = "4h_Donchian_EMA_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_high = high_roll.values
    donch_low = low_roll.values
    donch_mid = (donch_high + donch_low) / 2
    
    # 12h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h data for EMA50 (ensure alignment with completed 12h bars)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h closes
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (waits for 12h bar close)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, above 12h EMA50, and volume spike
            if (close[i] > donch_high[i] and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, below 12h EMA50, and volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian mid OR EMA turns bearish
            if (close[i] < donch_mid[i]) or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian mid OR EMA turns bullish
            if (close[i] > donch_mid[i]) or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals