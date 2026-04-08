#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend and Volume Confirmation (V2)
Hypothesis: Price breaking above/below 20-period Donchian channel on 4h timeframe,
filtered by 1d EMA trend direction and volume spikes, captures strong momentum moves
while avoiding false breakouts in ranging markets. Tightened volume filter and
added minimum holding period to reduce trades from ~100 to target 50-100/year.
Works in bull markets via breakout momentum and in bear markets via trend-filtered
mean reversion off channel boundaries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_tight_volume_v2"
timeframe = "4h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian Channel (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    
    # Align Donchian levels to avoid look-ahead
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    
    # Volume filter: current volume > 2.2x 24-period average (tighter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.2)
    
    # Minimum holding period: 3 bars (12 hours) to prevent whipsaw
    hold_count = 0
    min_hold = 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            hold_count = 0
            continue
        
        # Decrement hold counter
        if hold_count > 0:
            hold_count -= 1
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend reverses
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
                hold_count = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend reverses
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
                hold_count = 0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry (only if hold period elapsed)
            if hold_count == 0:
                # Trend filter: price vs 1d EMA50
                uptrend = close[i] > ema_50_1d_aligned[i]
                downtrend = close[i] < ema_50_1d_aligned[i]
                
                # Long: price breaks above Donchian upper + uptrend + volume spike
                if (close[i] > donchian_upper_aligned[i] and 
                    uptrend and 
                    vol_spike[i]):
                    position = 1
                    signals[i] = 0.25
                    hold_count = min_hold
                # Short: price breaks below Donchian lower + downtrend + volume spike
                elif (close[i] < donchian_lower_aligned[i] and 
                      downtrend and 
                      vol_spike[i]):
                    position = -1
                    signals[i] = -0.25
                    hold_count = min_hold
    
    return signals