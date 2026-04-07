#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Weekly Trend Filter and Volume Spike
# Hypothesis: Donchian(20) breakouts on 12h capture momentum bursts.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume spikes confirm institutional participation, reducing false breakouts.
# Works in bull markets via upward breakouts + uptrend, in bear via downward breakouts + downtrend.
# Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe.

name = "12h_donchian_breakout_weekly_trend_volume_v1"
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
    
    # Get weekly data for trend filter (using 1w as specified)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(40) for trend filter
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Daily data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period high/low)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: volume > 2.5x 30-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend turns bearish
            if close[i] < low_20_aligned[i] or close[i] < ema_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend turns bullish
            if close[i] > high_20_aligned[i] or close[i] > ema_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout + uptrend filter
                if close[i] > high_20_aligned[i] and close[i] > ema_40_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish breakout + downtrend filter
                elif close[i] < low_20_aligned[i] and close[i] < ema_40_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals