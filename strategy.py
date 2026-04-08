#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_atr_stop
Hypothesis: Donchian breakout in direction of 1d trend with volume confirmation and ATR trailing stop.
- Primary: 4h Donchian breakout (20-period) for entry
- Trend filter: 1d EMA(50) - only long when price > EMA, short when price < EMA
- Volume confirmation: 4h volume > 1.5x 20-period average volume
- Exit: ATR trailing stop (2.0) or Donchian breakout in opposite direction
- Position sizing: 0.25 for long, -0.25 for short
Target: 20-50 trades/year (80-200 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_atr_stop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Parameters
    position_size = 0.25
    atr_multiple = 2.0
    volume_multiple = 1.5
    
    # Start after warmup (max of Donchian, ATR, EMA periods)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions
            if (close[i] < (highest_since_entry - atr_multiple * atr[i])) or \
               (close[i] < donchian_low[i]) or \
               (close[i] < trend_1d_aligned[i]):
                position = 0
                highest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions
            if (close[i] > (lowest_since_entry + atr_multiple * atr[i])) or \
               (close[i] > donchian_high[i]) or \
               (close[i] > trend_1d_aligned[i]):
                position = 0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat, look for entry
            # Long entry: Donchian breakout up + 1d uptrend + volume spike
            if (close[i] > donchian_high[i]) and \
               (close[i] > trend_1d_aligned[i]) and \
               (volume[i] > vol_ma[i] * volume_multiple):
                position = 1
                highest_since_entry = high[i]
                signals[i] = position_size
            # Short entry: Donchian breakdown down + 1d downtrend + volume spike
            elif (close[i] < donchian_low[i]) and \
                 (close[i] < trend_1d_aligned[i]) and \
                 (volume[i] > vol_ma[i] * volume_multiple):
                position = -1
                lowest_since_entry = low[i]
                signals[i] = -position_size
    
    return signals