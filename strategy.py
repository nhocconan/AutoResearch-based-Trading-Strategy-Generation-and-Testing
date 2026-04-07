#!/usr/bin/env python3
"""
4h_atr_breakout_1d_trend_volume_v3
Hypothesis: ATR-based breakouts from 1-day high/low with volume confirmation and 1-day trend filter.
Trades breakouts in direction of daily trend to capture momentum while avoiding counter-trend whipsaws.
Works in both bull and bear markets by using volatility-based breakouts and trend alignment.
Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for breakout levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) for breakout threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily high/low for breakout levels (using previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Daily EMA for trend filter (20-period)
    ema_20 = df_1d['close'].ewm(span=20, adjust=False).mean().values
    
    # Align all daily data to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            vol_ma[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below daily low or trend turns bearish
            if close[i] < prev_low_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above daily high or trend turns bullish
            if close[i] > prev_high_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above daily high with volume and bullish trend
            if (close[i] > prev_high_aligned[i] + 0.5 * atr[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below daily low with volume and bearish trend
            elif (close[i] < prev_low_aligned[i] - 0.5 * atr[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals