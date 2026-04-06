# I’m going to implement a 1-day strategy that uses a 1-week high/low breakout with volume confirmation and a simple trend filter.
# The idea: In both bull and bear markets, price often continues in the direction of a breakout of the prior week’s range when accompanied by above-average volume.
# We use the weekly high and low as support/resistance levels. A break above the weekly high with volume > 1.5x the 20-day average volume triggers a long.
# A break below the weekly low with volume > 1.5x the 20-day average volume triggers a short.
# We exit when price returns to the middle of the weekly range (or opposite breakout occurs) to avoid giving back too much profit.
# Position size is 0.25 (25% of capital) to keep drawdowns manageable.
# This should produce a moderate number of trades (target 50-150 over 4 years) because breakouts are infrequent but significant.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_breakout_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # need enough data for weekly lookback and volume average
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data (1w) for high/low of the prior week
    df_1w = get_htf_data(prices, '1w')
    # Weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    # Align to daily timeframe (shifted by 1 week to avoid look-ahead)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: volume > 1.5x 20-day average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after volume MA warmup
        # Skip if required data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below weekly low OR returns to midpoint of weekly range
            midpoint = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2.0
            if close[i] < weekly_low_aligned[i] or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above weekly high OR returns to midpoint
            midpoint = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2.0
            if close[i] > weekly_high_aligned[i] or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: price breaks weekly high/low with volume confirmation
            if volume[i] > volume_threshold[i]:
                if close[i] > weekly_high_aligned[i]:
                    # Break above weekly high - bullish
                    signals[i] = 0.25
                    position = 1
                elif close[i] < weekly_low_aligned[i]:
                    # Break below weekly low - bearish
                    signals[i] = -0.25
                    position = -1
    
    return signals