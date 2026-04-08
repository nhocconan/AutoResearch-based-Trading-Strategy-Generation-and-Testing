#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with weekly trend filter and volume confirmation
# Uses weekly EMA50 for trend direction (bullish when price > EMA50, bearish when price < EMA50)
# Enters long when price breaks above 12h Donchian upper channel (20-period) in uptrend
# Enters short when price breaks below 12h Donchian lower channel (20-period) in downtrend
# Requires volume > 1.5x 20-period average for confirmation
# Exits when price crosses back through the Donchian middle (10-period average of high/low)
# Designed to capture strong trends while avoiding whipsaw in choppy markets
# Target: 50-150 total trades over 4 years via strict breakout + trend + volume confluence

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle channel: average of 20-period high and low
    channel_middle = (high_max_20 + low_min_20) / 2
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50  # Need EMA and Donchian buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly values for current 12h bar
        ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)[i]
        
        # Trend filter: bullish when price > weekly EMA50, bearish when price < weekly EMA50
        bullish_trend = close[i] > ema50_1w_aligned
        bearish_trend = close[i] < ema50_1w_aligned
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below channel middle
            if close[i] < channel_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above channel middle
            if close[i] > channel_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_confirm:
                # Long entry: price breaks above upper Donchian channel in uptrend
                if bullish_trend and close[i] > high_max_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below lower Donchian channel in downtrend
                elif bearish_trend and close[i] < low_min_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals