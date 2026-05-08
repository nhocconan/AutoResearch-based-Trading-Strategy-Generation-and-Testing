#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# The Donchian channel identifies breakout points with clear support/resistance.
# We enter on breakout of the 20-period channel in the direction of the 1d EMA50 trend,
# confirmed by volume > 1.5x 20-period average. This combines trend-following with
# momentum confirmation to avoid whipsaws. Exit on opposite channel touch or trend reversal.
# Targets 20-50 trades per year (~80-200 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel: 20-period high and low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high_roll[i]
        low_val = low_roll[i]
        ema50_val = ema50_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band, above 1d EMA50, volume confirmation
            if close_val > high_val and close_val > ema50_val and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band, below 1d EMA50, volume confirmation
            elif close_val < low_val and close_val < ema50_val and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches lower Donchian band or closes below 1d EMA50
            if close_val < low_val or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches upper Donchian band or closes above 1d EMA50
            if close_val > high_val or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals