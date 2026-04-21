#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ATRFilter_v1
Hypothesis: Buy breakouts above 20-period Donchian high with volume confirmation and 12h trend filter, sell breakdowns below 20-period low. Works in bull (breakouts continue) and bear (breakdowns continue). Uses ATR-based stoploss to limit drawdown. Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period Donchian channels on primary timeframe (4h)
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper_channel = high_roll.values
    lower_channel = low_roll.values
    
    # ATR for stoploss and volume filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume_ok[i]
        
        if position == 0:
            # Long: price breaks above upper channel + 12h uptrend + volume
            if (price > upper_channel[i] and 
                ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] and
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel + 12h downtrend + volume
            elif (price < lower_channel[i] and 
                  ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] and
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 12h EMA34 (trend change) or ATR-based stop
            if (price < ema_34_12h_aligned[i] or 
                price < prices['close'].iloc[i-1] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 12h EMA34 (trend change) or ATR-based stop
            if (price > ema_34_12h_aligned[i] or 
                price > prices['close'].iloc[i-1] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0