#!/usr/bin/env python3
# 12H_DONCHIAN_BREAKOUT_1D_VOLUME_CONFIRMATION
# Hypothesis: Donchian breakout on 12h captures medium-term momentum. 1-day volume surge (above 1.5x 20-period average) confirms institutional participation.
# In 1d uptrend (price > EMA50), go long on 12h Donchian(20) breakout with volume confirmation.
# In 1d downtrend (price < EMA50), go short on 12h Donchian(20) breakdown with volume confirmation.
# Works in both bull and bear markets: 1d EMA50 filter avoids counter-trend trades, Donchian breakout captures momentum within trend.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_DONCHIAN_BREAKOUT_1D_VOLUME_CONFIRMATION"
timeframe = "12h"
leverage = 1.0

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
    
    # Daily data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    # Current volume
    vol_current = df_1d['volume'].values
    # Volume confirmation: current > 1.5 * average
    vol_confirm = vol_current > (1.5 * vol_avg)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    vol_current_aligned = align_htf_to_ltf(prices, df_1d, vol_current)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(vol_current_aligned[i]) or np.isnan(vol_confirm_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + 12h Donchian breakout + volume confirmation
            if (close[i] > ema50_aligned[i] and 
                high[i] > highest_high[i - 1] and  # Current high breaks above previous Donchian high
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + 12h Donchian breakdown + volume confirmation
            elif (close[i] < ema50_aligned[i] and 
                  low[i] < lowest_low[i - 1] and  # Current low breaks below previous Donchian low
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or breakdown below Donchian low
            if (close[i] <= ema50_aligned[i] or 
                low[i] < lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or breakout above Donchian high
            if (close[i] >= ema50_aligned[i] or 
                high[i] > highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals