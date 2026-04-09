#!/usr/bin/env python3
# 12h_daily_ema_volume_filter_v1
# Hypothesis: 12h strategy using 1d EMA trend with volume confirmation.
# Long when price > EMA50_1d and volume > 1.5x 20-period average.
# Short when price < EMA50_1d and volume > 1.5x 20-period average.
# Exit on opposite EMA cross or volume drop.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong trends in both bull and bear markets with volume confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_ema_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price closes below EMA or volume drops
            if close[i] < ema_50_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above EMA or volume drops
            if close[i] > ema_50_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry with volume confirmation
            bullish_entry = (close[i] > ema_50_aligned[i]) and volume_confirmed
            bearish_entry = (close[i] < ema_50_aligned[i]) and volume_confirmed
            
            if bullish_entry:
                position = 1
                signals[i] = 0.25
            elif bearish_entry:
                position = -1
                signals[i] = -0.25
    
    return signals