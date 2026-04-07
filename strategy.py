#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
# Hypothesis: 12h Donchian(20) breakouts in direction of 1d EMA(50) trend with volume confirmation capture strong moves while avoiding false breakouts in ranging markets.
# Uses 1d EMA for trend filter (works in bull/bear) and volume spike for confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for EMA trend and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Calculate average volume on 1d
    avg_volume = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA and average volume to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    
    # Donchian(20) on 12h high/low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint of Donchian channel or trend changes
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint of Donchian channel or trend changes
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirmed = volume[i] > 1.5 * avg_volume_aligned[i]
            
            if volume_confirmed:
                # Long breakout: price breaks above 20-period high in uptrend
                if close[i] > highest_high[i] and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below 20-period low in downtrend
                elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals