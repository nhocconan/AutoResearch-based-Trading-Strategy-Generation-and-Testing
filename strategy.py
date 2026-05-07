#!/usr/bin/env python3
# 1D_WeeklyTrend_VolumeSpike
# Hypothesis: Uses weekly trend filter (EMA10 on 1w close) with daily price action (close > open) and volume spike (2x 20-day average) for entries.
# Exits when trend weakens (price < weekly EMA10) or volume drops. Designed for low frequency (<25/year) and strong performance in both bull and bear regimes.
# Target: 10-20 trades per year per symbol with clear trend-following logic.

name = "1D_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Daily volume filter: current volume > 2.0x average volume (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure we have volume MA and weekly data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema10_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Open (bullish daily candle) + price > weekly EMA10 + volume spike
            if (close[i] > prices['open'].iloc[i] and 
                close[i] > ema10_1w_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Close < Open (bearish daily candle) + price < weekly EMA10 + volume spike
            elif (close[i] < prices['open'].iloc[i] and 
                  close[i] < ema10_1w_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses weekly EMA10 against position OR volume drops below average
            trend_exit = (position == 1 and close[i] < ema10_1w_aligned[i]) or \
                         (position == -1 and close[i] > ema10_1w_aligned[i])
            volume_exit = volume[i] < vol_ma[i]  # exit on low volume
            
            if trend_exit or volume_exit:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals