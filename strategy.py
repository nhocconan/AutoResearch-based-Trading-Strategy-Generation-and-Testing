#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Index with 1d trend filter and volume confirmation
# Hypothesis: Elder Ray measures bull/bear power relative to EMA13. Combined with 1d trend filter
# (price above/below EMA50) and volume confirmation, it captures strong directional moves
# while avoiding chop. Works in bull via bull power signals, in bear via bear power signals.
# Target: 12-37 trades/year to minimize fee drag.
name = "6h_elder_ray_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # Get daily data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (6-period EMA approximation for 6h)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema50_1d = daily_close_s.ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily volume moving average for confirmation
    daily_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bear power becomes positive (bulls losing control) OR trend fails
            if bear_power[i] > 0 or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: bull power becomes negative (bears losing control) OR trend fails
            if bull_power[i] < 0 or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: bull power positive (bulls in control) + uptrend + volume confirmation
            if bull_power[i] > 0 and close[i] > ema50_1d_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: bear power negative (bears in control) + downtrend + volume confirmation
            elif bear_power[i] < 0 and close[i] < ema50_1d_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals