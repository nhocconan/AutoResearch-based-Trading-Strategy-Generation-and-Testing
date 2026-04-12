#!/usr/bin/env python3
"""
6h_1d_Keltner_Breakout_Volume_Confirmation_v1
Hypothesis: On 6h timeframe, buy when price breaks above Keltner upper band (EMA20 + 2*ATR) with volume confirmation and 1d trend filter (price > 1d EMA50), sell when price breaks below Keltner lower band with volume confirmation and 1d trend filter (price < 1d EMA50). Uses Keltner channels for volatility-adaptive breakouts, volume confirmation for strength, and 1d EMA50 for trend alignment. Designed for 12-30 trades/year by requiring multiple confluence factors.
Works in bull markets via long breakouts and in bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Keltner_Breakout_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.where(high-low > 0, high-low, 0.0001)).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Keltner breakout conditions
        breakout_up = close[i] > keltner_upper[i-1]  # Break above previous period's upper band
        breakout_down = close[i] < keltner_lower[i-1]  # Break below previous period's lower band
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter from 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_spike and above_ema
        short_entry = breakout_down and volume_spike and below_ema
        
        # Exit conditions: price returns to EMA20 (middle of Keltner)
        long_exit = close[i] < ema_20[i]
        short_exit = close[i] > ema_20[i]
        
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