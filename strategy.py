#!/usr/bin/env python3
# 1d_weekly_fractal_breakout_volume_v1
# Hypothesis: Daily price breaks above/below weekly Williams fractal levels with volume confirmation.
# Fractals identify significant swing highs/lows; breakouts indicate trend continuation.
# Volume filter ensures institutional participation. Works in bull/bear by following weekly structure.
# Target: 15-25 trades/year via weekly fractal breakouts + volume confirmation.

name = "1d_weekly_fractal_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for fractal calculation
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate Williams fractals on weekly data
    bearish_fractal, bullish_fractal = compute_williams_fractals(weekly_high, weekly_low)
    
    # Need 2-bar confirmation for fractals (they form after 2 bars close)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_weekly, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_weekly, bullish_fractal, additional_delay_bars=2)
    
    # Daily volume filter: 20-period average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 20  # For volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below weekly bullish fractal (support)
            if close[i] < bullish_fractal_confirmed[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above weekly bearish fractal (resistance)
            if close[i] > bearish_fractal_confirmed[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly bearish fractal (resistance) with volume
            if (close[i] > bearish_fractal_confirmed[i] and 
                volume_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly bullish fractal (support) with volume
            elif (close[i] < bullish_fractal_confirmed[i] and 
                  volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals