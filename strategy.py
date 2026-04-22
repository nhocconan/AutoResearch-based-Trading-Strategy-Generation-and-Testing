#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout (20) with 1d trend filter and volume confirmation
    # Donchian channels provide clear breakout levels with built-in trend following
    # 1d EMA50 filters for multi-day trend direction to avoid counter-trend trades
    # Volume confirmation (1.5x 20-period MA) ensures institutional participation
    # Works in both bull and bear markets by following the dominant trend
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian Channel (20-period) on 12h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian + volume confirmation + price above EMA50 (uptrend)
            if close[i] > high_max[i] and vol_confirm[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian + volume confirmation + price below EMA50 (downtrend)
            elif close[i] < low_min[i] and vol_confirm[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level or trend reversal
            if position == 1:
                if close[i] < low_min[i]:  # Break below lower band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_max[i]:  # Break above upper band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0