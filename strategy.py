#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume spike confirmation.
# Long: Price breaks above Donchian high + 12h EMA34 uptrend + volume > 2x 20-period average.
# Short: Price breaks below Donchian low + 12h EMA34 downtrend + volume > 2x 20-period average.
# Exit: When price crosses back through Donchian midpoint (mean reversion) or trend reverses.
# Uses price channel structure for clear entries/exits, volume for confirmation, EMA34 for trend filter.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Calculate EMA34 for 12h trend
    close_12h = pd.Series(df_12h['close'].values)
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA34 slope
        ema34_prev = ema34_12h_aligned[i-1] if i > 0 else ema34_12h_aligned[i]
        uptrend = ema34_12h_aligned[i] > ema34_prev
        downtrend = ema34_12h_aligned[i] < ema34_prev
        
        if position == 0:
            # Long: break above Donchian high + uptrend + volume spike
            if high[i] > high_roll[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + downtrend + volume spike
            elif low[i] < low_roll[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint OR trend reverses
            if close[i] < donchian_mid[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint OR trend reverses
            if close[i] > donchian_mid[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals