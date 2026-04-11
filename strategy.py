#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_v1
# Strategy: Daily Camarilla pivot breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Weekly trend filters reduce false breakouts in sideways markets. Combined with daily Camarilla breakouts and volume confirmation, this strategy captures strong directional moves while avoiding chop. Designed for low trade frequency (15-25/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily Camarilla levels from previous day
    prev_close = prices['close'].shift(1).values
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    
    # Camarilla levels
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    H4 = prev_close + 1.1 * rng / 2
    L4 = prev_close - 1.1 * rng / 2
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(H4[i]) or np.isnan(L4[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals using Camarilla levels
        breakout_up = high[i] > H3[i-1]
        breakdown_down = low[i] < L3[i-1]
        
        # Weekly EMA200 trend filter
        trend_bullish = close[i] > ema_200_1w_aligned[i]
        trend_bearish = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using stronger H4/L4 levels
        elif position == 1 and low[i] < L4[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H4[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals