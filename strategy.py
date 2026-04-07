#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian(20) Breakout + Weekly EMA(50) Trend + Volume Filter
# Hypothesis: Breakouts above/below 20-day high/low in direction of weekly trend with volume confirmation
# work in both bull and bear markets by capturing sustained moves. Weekly trend filter avoids counter-trend trades.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_donchian20_weekly_ema_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Donchian Channels (20-period)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Weekly EMA(50) for trend filter
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient data
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: require volume > 1.5x average
        volume_ok = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA(50) or Donchian lower
            if close[i] < ema_50_weekly_aligned[i] or close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA(50) or Donchian upper
            if close[i] > ema_50_weekly_aligned[i] or close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Breakout above upper band with uptrend
                if close[i] > upper[i] and close[i] > ema_50_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower band with downtrend
                elif close[i] < lower[i] and close[i] < ema_50_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals