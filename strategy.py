#!/usr/bin/env python3
"""
1h 4H/1D Volatility Breakout with Volume Confirmation
Hypothesis: In both bull and bear markets, volatility bursts (ATR expansion)
combined with volume spikes often precede sustained directional moves.
Use 4H trend for direction, 1H for entry timing via volatility/volume breakout.
Volume confirmation filters false breakouts. Designed for low trade frequency.
"""
name = "1h_VolVol_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H TREND (EMA 21) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_4h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1H ATR (21) FOR VOLATILITY BREAKOUT ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # === 1H VOLUME (20) SPIKE ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # High threshold for rarity
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_4h[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend + volatility breakout up + volume spike
            if (close[i] > trend_4h[i] and 
                close[i] > close[i-1] + 0.5 * atr[i] and
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Downtrend + volatility breakout down + volume spike
            elif (close[i] < trend_4h[i] and 
                  close[i] < close[i-1] - 0.5 * atr[i] and
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Close below 4H trend OR volatility contraction
            if close[i] < trend_4h[i] or (high[i] - low[i]) < 0.3 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close above 4H trend OR volatility contraction
            if close[i] > trend_4h[i] or (high[i] - low[i]) < 0.3 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals