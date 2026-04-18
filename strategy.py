#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Volume_Trend_Breakout_With_ATRStop
Hypothesis: 4h price breaks above/below Camarilla R1/S1 levels with volume spike and trend confirmation.
In bull markets, captures breakouts above R1; in bear markets, captures breakdowns below S1.
Trend filter uses 1d EMA34 to avoid counter-trend trades. Volume spike ensures momentum confirmation.
Designed for 15-25 trades/year to minimize fee drift while capturing strong directional moves.
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
    
    # ATR for ATR stop and Camarilla
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Camarilla levels from previous day (using daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 4h bar using previous day's range
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + range_1d * 1.1 / 12
    camarilla_s1 = close_1d - range_1d * 1.1 / 12
    
    # Align to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Trend filter: 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # ATR stop or trend reversal
            if price < r1 - 1.5 * atr_val:  # ATR stop
                signals[i] = 0.0
                position = 0
            elif price < ema34:  # trend reversal
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # ATR stop or trend reversal
            if price > s1 + 1.5 * atr_val:  # ATR stop
                signals[i] = 0.0
                position = 0
            elif price > ema34:  # trend reversal
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Volume_Trend_Breakout_With_ATRStop"
timeframe = "4h"
leverage = 1.0