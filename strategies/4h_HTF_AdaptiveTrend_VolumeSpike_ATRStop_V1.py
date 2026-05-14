#!/usr/bin/env python3
"""
4h_HTF_AdaptiveTrend_VolumeSpike_ATRStop_V1
Hypothesis: Use 12h EMA34 trend filter + 4h Donchian(20) breakout with volume spike (>2x 20-bar MA) and ATR(14) stoploss (2.0x). 12h EMA34 reduces whipsaw in sideways markets, volume spike confirms breakout legitimacy, ATR stop manages risk. Designed to work in both bull (catch trends) and bear (avoid false breaks via 12h filter) markets. Target 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')  # for EMA34 trend filter
    
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h EMA34 for Trend Filter ===
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above Donchian(20) high with volume spike and 12h uptrend
            if price > highest_20[i-1] and vol_ok and ema_12h_aligned[i] < price:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian(20) low with volume spike and 12h downtrend
            elif price < lowest_20[i-1] and vol_ok and ema_12h_aligned[i] > price:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < highest_20[i-1] - 2.0 * atr[i] or (price < lowest_20[i-1] and vol_ok and ema_12h_aligned[i] > price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > lowest_20[i-1] + 2.0 * atr[i] or (price > highest_20[i-1] and vol_ok and ema_12h_aligned[i] < price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_AdaptiveTrend_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0