#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) Breakout + Daily EMA(50) Trend + Volume Spike
# Hypothesis: Breakouts from 20-period Donchian channel with volume confirmation
# and aligned with daily trend capture strong momentum moves while avoiding whipsaws.
# Works in bull markets via long breakouts and bear markets via short breakdowns.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_donchian_breakout_daily_trend_v1"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    
    # Donchian Channel (20) on 4h
    dc_period = 20
    upper = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Daily EMA(50) for trend filter
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(dc_period, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_daily_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend changes to down
            if close[i] < lower[i] or close[i] < ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend changes to up
            if close[i] > upper[i] or close[i] > ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above upper channel with uptrend
                if close[i] > upper[i] and close[i] > ema_50_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower channel with downtrend
                elif close[i] < lower[i] and close[i] < ema_50_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals