#!/usr/bin/env python3
"""
6h_12h_Donchian_Breakout_with_Volume_and_1dTrend
Hypothesis: 6s Donchian(20) breakouts filtered by 12h trend (EMA34) and volume confirmation.
Only take long when price breaks above 6h Donchian high, 12h EMA34 is rising, and volume > 1.5x average.
Short when price breaks below 6h Donchian low, 12h EMA34 is falling, and volume > 1.5x average.
Uses ATR-based volatility filter to avoid choppy markets.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend direction
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h EMA34 slope for trend strength (rising/falling)
    ema_slope = np.diff(ema_34_12h_aligned, prepend=ema_34_12h_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volatility filter: ATR(20) to avoid choppy markets
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(np.roll(close, 1) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_20[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid extreme volatility
        vol_ma_long = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values
        vol_filter = (not np.isnan(vol_ma_long[i])) and (atr_20[i] < vol_ma_long[i] * 2)
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volume
            if (close[i] > donchian_high[i] and ema_rising[i] and 
                vol_confirm and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with downtrend and volume
            elif (close[i] < donchian_low[i] and ema_falling[i] and 
                  vol_confirm and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below Donchian low or trend changes
            if close[i] < donchian_low[i] or not ema_rising[i] or not vol_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above Donchian high or trend changes
            if close[i] > donchian_high[i] or not ema_falling[i] or not vol_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_Donchian_Breakout_with_Volume_and_1dTrend"
timeframe = "6h"
leverage = 1.0