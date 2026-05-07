#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Breakouts above Camarilla R1 or below S1 on 12h, confirmed by 1-day EMA50 trend and volume spikes, capture directional moves in trending markets. Works in bull (R1 breakouts in uptrend) and bear (S1 breakdowns in downtrend). Low-frequency signals via 12h timeframe and confluence of Camarilla levels, trend, and volume.
"""
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior 12h bar
    # R1 = close + (high - low) * 1.12 / 12
    # S1 = close - (high - low) * 1.12 / 12
    # We use the prior bar's levels to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    camarilla_width = (prior_high - prior_low) * 1.12 / 12.0
    r1 = prior_close + camarilla_width
    s1 = prior_close - camarilla_width
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 2.0 * 30-period average (stricter for lower frequency)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close breaks above R1 + 1d uptrend + volume
            if close[i] > r1[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1 + 1d downtrend + volume
            elif close[i] < s1[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the broken level
            if position == 1:
                if close[i] < r1[i]:  # Exit long if price goes back below R1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > s1[i]:  # Exit short if price goes back above S1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals