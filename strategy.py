#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1d Donchian channels for structure, 1w EMA50 for trend filter, and volume spike for confirmation.
# Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns.
# The 1w EMA50 filter ensures we only trade with the higher timeframe trend, reducing whipsaws.

name = "1d_Donchian20_1wEMA50_VolumeSpike_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_shifted = np.roll(ema50_1w, 1)
    ema50_1w_shifted[0] = np.nan
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_shifted)
    
    # Calculate 1d Donchian channels (20-period) from prior completed 1d bar
    # We need 20 periods of high/low to calculate the channel
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 1d bar levels (avoid look-ahead)
    highest_high_shifted = np.roll(highest_high, 1)
    lowest_low_shifted = np.roll(lowest_low, 1)
    highest_high_shifted[0] = np.nan
    lowest_low_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(highest_high_shifted[i]) or np.isnan(lowest_low_shifted[i]) or
            np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
        vol_ema_20 = pd.Series(volume[:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        
        if position == 0:
            # Long conditions: break above Donchian upper band AND 1w EMA50 uptrend AND volume spike
            if close[i] > highest_high_shifted[i] and close[i] > ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower band AND 1w EMA50 downtrend AND volume spike
            elif close[i] < lowest_low_shifted[i] and close[i] < ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band OR below 1w EMA50
            if close[i] < lowest_low_shifted[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band OR above 1w EMA50
            if close[i] > highest_high_shifted[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals