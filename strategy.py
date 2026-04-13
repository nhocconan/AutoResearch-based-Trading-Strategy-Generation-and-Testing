#!/usr/bin/env python3
"""
4h_12h_Chaikin_Money_Flow_Volume_Signal_Strategy
Hypothesis: Chaikin Money Flow (CMF) measures buying/selling pressure with volume-weighted accumulation/distribution.
CMF > 0 indicates buying pressure, CMF < 0 indicates selling pressure. Combined with 12h trend filter (EMA25) and volume expansion,
this captures sustained institutional flow while avoiding whipsaws. Works in bull (strong CMF>0) and bear (strong CMF<0) markets.
Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for CMF and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Chaikin Money Flow (CMF) on 12h data
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high_12h - low_12h
    hl_range = np.where(hl_range == 0, 1, hl_range)  # replace zeros with 1 to avoid div by zero
    mf_multiplier = ((close_12h - low_12h) - (high_12h - close_12h)) / hl_range
    
    # Money Flow Volume = Money Flow Multiplier * Volume
    mf_volume = mf_multiplier * volume_12h
    
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum()
    volume_sum = pd.Series(volume_12h).rolling(window=20, min_periods=20).sum()
    cmf = np.where(volume_sum != 0, mf_volume_sum / volume_sum, 0)
    
    # 12h EMA25 for trend filter
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Volume expansion on 12h: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume_12h > (vol_ma_20 * 1.5)
    
    # Align CMF, EMA25, and volume expansion to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_12h, cmf)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_12h, volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema_25_12h_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. CMF > 0 (buying pressure)
        # 2. Price above 12h EMA25 (12h trend filter)
        # 3. Volume expansion
        long_condition = (cmf_aligned[i] > 0) and (close[i] > ema_25_12h_aligned[i]) and (volume_expansion_aligned[i] > 0.5)
        
        # Short conditions:
        # 1. CMF < 0 (selling pressure)
        # 2. Price below 12h EMA25 (12h trend filter)
        # 3. Volume expansion
        short_condition = (cmf_aligned[i] < 0) and (close[i] < ema_25_12h_aligned[i]) and (volume_expansion_aligned[i] > 0.5)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_12h_Chaikin_Money_Flow_Volume_Signal_Strategy"
timeframe = "4h"
leverage = 1.0