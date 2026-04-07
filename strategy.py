#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume and Volume Spike Confirmation
# Hypothesis: Weekly Donchian(20) breakouts on 1d timeframe, confirmed by volume spike and 1w EMA trend,
# capture strong momentum moves in both bull and bear markets. Weekly EMA filter ensures we only
# trade in the direction of the higher timeframe trend, reducing false breakouts.
# Volume spike ensures breakouts have institutional participation. Target: 15-25 trades/year.

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA(20) on weekly close
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    
    # Align weekly EMA to daily
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of past 20 days
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume spike detection: current volume > 2.0 * 20-day average volume
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend changes
            if close[i] < lowest_low[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend changes
            if close[i] > highest_high[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper band + volume spike + uptrend
            if close[i] > highest_high[i] and volume_spike[i] and close[i] > ema_20_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band + volume spike + downtrend
            elif close[i] < lowest_low[i] and volume_spike[i] and close[i] < ema_20_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals