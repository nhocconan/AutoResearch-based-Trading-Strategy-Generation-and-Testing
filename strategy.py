#!/usr/bin/env python3
"""
Hypothesis: Combine 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high AND 12h EMA34 > prior close AND volume > 1.5x 20-bar avg.
Short when price breaks below 20-period Donchian low AND 12h EMA34 < prior close AND volume > 1.5x 20-bar avg.
Exit on opposite Donchian break or volume drop below average.
Uses discrete position size 0.25 to limit drawdown. Targets 20-40 trades/year (~80-160 total).
Works in bull via trend-following breaks, in bear via short breakdowns with trend filter.
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
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for Donchian(20) and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, 12h EMA up, volume spike
            if (price > donch_high[i] and 
                ema_12h_aligned[i] > close[i-1] and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, 12h EMA down, volume spike
            elif (price < donch_low[i] and 
                  ema_12h_aligned[i] < close[i-1] and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR volume drops below average
            if price < donch_low[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR volume drops below average
            if price > donch_high[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0