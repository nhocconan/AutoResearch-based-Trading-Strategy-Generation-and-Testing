#!/usr/bin/env python3
"""
6h_TP_DualBarrier_VolumeConfirmation
Hypothesis: Uses a dual-barrier system (ATR-based upper/lower bands) on 6h timeframe
to capture breakouts with volume confirmation, filtered by 12h trend direction.
The dual barriers prevent whipsaws in ranging markets while capturing strong moves.
Volume confirmation ensures breakouts have conviction. Designed for 12-37 trades/year.
"""

name = "6h_TP_DualBarrier_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

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
    
    # ATR for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Dual barriers: upper and lower bands based on ATR
    mult = 2.5
    upper_band = close + (atr * mult)
    lower_band = close - (atr * mult)
    
    # 12h trend filter: EMA of daily close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band with volume confirmation and 12h uptrend
            if (close[i] > upper_band[i] and 
                volume[i] > vol_ma[i] and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume confirmation and 12h downtrend
            elif (close[i] < lower_band[i] and 
                  volume[i] > vol_ma[i] and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to middle (close) or reverses below lower band
            if close[i] < close[i-1] or close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to middle (close) or reverses above upper band
            if close[i] > close[i-1] or close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals