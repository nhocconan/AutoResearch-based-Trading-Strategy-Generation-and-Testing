#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) with 1-day EMA trend filter and volume confirmation.
- Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
- Enter long when Bull Power > 0, Bear Power < 0, and price above 1-day EMA50 with volume > 1.5x 20-period volume MA
- Enter short when Bull Power < 0, Bear Power > 0, and price below 1-day EMA50 with volume > 1.5x 20-period volume MA
- Exit when Elder Ray signals reverse (Bull Power and Bear Power cross zero)
- Fixed position size 0.25 to manage drawdown
- Uses 1-day trend filter to avoid counter-trend trades
- Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
"""

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
    
    # Get 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (using 13-period EMA on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA and EMA13
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        bull = bull_power[i]
        bear = bear_power[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Look for Elder Ray signals with volume confirmation and trend filter
            # Long: Bull Power > 0, Bear Power < 0, price above EMA50, volume spike
            if bull > 0 and bear < 0 and price > ema_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price below EMA50, volume spike
            elif bull < 0 and bear > 0 and price < ema_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Elder Ray turns bearish (Bull Power < 0 or Bear Power > 0)
            if bull < 0 or bear > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Elder Ray turns bullish (Bull Power > 0 or Bear Power < 0)
            if bull > 0 or bear < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Volume_1dEMA50"
timeframe = "6h"
leverage = 1.0