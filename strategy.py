#!/usr/bin/env python3
"""
4h_atr_breakout_12h_trend_volume_v2
Hypothesis: ATR breakout from 20-bar range with 12h EMA trend filter and volume confirmation.
Breakouts above/below 20-period high/low with volume spike and aligned 12h trend capture
continuation moves in both bull and bear markets. ATR-based stop loss limits drawdown.
Target: 20-50 trades/year on 4h with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_12h_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr[i]) or
            np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema12h = close[i] > ema20_12h_aligned[i]
        below_ema12h = close[i] < ema20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: ATR-based stop or trend reversal with volume
            if close[i] <= (high_20[i] - 2.0 * atr[i]) or (below_ema12h and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stop or trend reversal with volume
            if close[i] >= (low_20[i] + 2.0 * atr[i]) or (above_ema12h and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout with volume and trend alignment
            if close[i] > high_20[i] and vol_spike and above_ema12h:
                # Bullish breakout with volume and uptrend
                position = 1
                signals[i] = 0.25
            elif close[i] < low_20[i] and vol_spike and below_ema12h:
                # Bearish breakout with volume and downtrend
                position = -1
                signals[i] = -0.25
    
    return signals