#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Donchian Breakout + Daily Volume Spike + 1d EMA Trend
# Hypothesis: Weekly Donchian channels (20-period) identify major support/resistance on 1w chart.
# Breakouts above weekly high with daily uptrend and volume spike signal long entries.
# Breakdowns below weekly low with daily downtrend and volume spike signal short entries.
# Works in bull markets via weekly high breakouts + uptrend, in bear via weekly low breakdowns + downtrend.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_weekly_donchian_1d_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    # Use previous week's data to avoid look-ahead
    weekly_high = df_1w['high'].rolling(window=20, min_periods=20).max().shift(1).values
    weekly_low = df_1w['low'].rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Get 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily trend filter: EMA(20) of daily close
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (high threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly low or trend turns bearish
            if close[i] < weekly_low_aligned[i] or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above weekly high or trend turns bullish
            if close[i] > weekly_high_aligned[i] or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above weekly high + uptrend
                if close[i] > weekly_high_aligned[i] and close[i] > ema_20_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below weekly low + downtrend
                elif close[i] < weekly_low_aligned[i] and close[i] < ema_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals