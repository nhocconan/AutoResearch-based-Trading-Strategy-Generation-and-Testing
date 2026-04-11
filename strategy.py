#!/usr/bin/env python3
"""
4h_1d_Volume_Spike_Keltner_Breakout_v1
Hypothesis: Uses 1-day Keltner channels with volume spikes to identify volatility expansions in trending markets.
Designed to work in both bull and bear markets by capturing volatility bursts that often precede strong moves.
Trades only when volatility expands (ATR-based) and volume confirms, reducing whipsaws and focusing on high-probability breakouts.
Targets 20-40 trades per year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Volume_Spike_Keltner_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Keltner channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 40-period EMA for 4h trend filter
    ema_40_4h = pd.Series(close).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Volume spike filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1-day Keltner channels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period EMA of close for Keltner center
    keltner_center_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True range for 1-day ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper_1d = keltner_center_1d + 2.0 * atr_1d
    keltner_lower_1d = keltner_center_1d - 2.0 * atr_1d
    
    # Align Keltner channels to 4h timeframe (wait for daily close)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    keltner_center_aligned = align_htf_to_ltf(prices, df_1d, keltner_center_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_40_4h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(keltner_center_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 2.0x 20-period average
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend filter: price above/below 40-period EMA
        uptrend = close[i] > ema_40_4h[i]
        downtrend = close[i] < ema_40_4h[i]
        
        # Breakout conditions using daily Keltner channels
        breakout_up = close[i] > keltner_upper_aligned[i]   # Break above upper Keltner
        breakdown_down = close[i] < keltner_lower_aligned[i] # Break below lower Keltner
        
        # Entry conditions: volatility expansion + volume spike + trend alignment
        long_entry = breakout_up and volume_spike and uptrend
        short_entry = breakdown_down and volume_spike and downtrend
        
        # Exit conditions: return to Keltner center or trend reversal
        long_exit = (close[i] < keltner_center_aligned[i]) or (not uptrend)
        short_exit = (close[i] > keltner_center_aligned[i]) or (not downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals