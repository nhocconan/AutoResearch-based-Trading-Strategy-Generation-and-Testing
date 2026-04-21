#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_VolumeATRFilter_V1
Hypothesis: Use 1d Camarilla R1/S1 breakout with volume spike (>2x 20-bar MA) and ATR(14) stop (1.5x). 1h timeframe for entry timing, 1d for signal direction. Volume confirms breakout legitimacy, ATR stop manages risk. Session filter (08-20 UTC) reduces noise. Designed for 15-30 trades/year per symbol with discrete sizing (0.20) to control fee drag and drawdown in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Points (based on prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels using prior day's OHLC
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pivot - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 1h: each 1d value applies to the following 24 bars (1h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (precomputed for speed)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0 and in_session:
            # Long: break above R1 with volume spike
            if price > r1_aligned[i-1] and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume spike
            elif price < s1_aligned[i-1] and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < r1_aligned[i-1] - 1.5 * atr[i] or (price < s1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > s1_aligned[i-1] + 1.5 * atr[i] or (price > r1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_VolumeATRFilter_V1"
timeframe = "1h"
leverage = 1.0