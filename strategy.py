#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout + 1d Trend + Volume Spike
# Hypothesis: Donchian breakouts capture trend continuation with low lag.
# Daily trend filter ensures alignment with higher-timeframe momentum.
# Volume spike confirms institutional participation in the breakout.
# Designed for 12h timeframe with low trade frequency (12-37/year).
# Works in bull via long breakouts + daily uptrend + volume,
# in bear via short breakouts + daily downtrend + volume.

name = "12h_donchian20_1d_volume_trend_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: EMA(50) of daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR daily trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR daily trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.28
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian high with daily uptrend
                if close[i] > high_20[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.28
                # Short: price breaks below Donchian low with daily downtrend
                elif close[i] < low_20[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.28
    
    return signals