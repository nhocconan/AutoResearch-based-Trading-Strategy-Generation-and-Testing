#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Donchian(20) breakouts on daily timeframe capture strong trends.
# Weekly EMA filter ensures alignment with higher-timeframe trend.
# Volume confirmation filters out false breakouts. Works in both bull and bear markets
# by following weekly trend. Target: 8-20 trades/year (32-80 total over 4 years).

name = "1d_donchian_breakout_1w_trend_v1"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Donchian Channel (20) on daily
    dc_period = 20
    
    # Upper band: highest high of last 20 periods
    upper = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    # Lower band: lowest low of last 20 periods
    lower = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Weekly EMA(20) for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(dc_period, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_20_weekly_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend changes to down
            if close[i] < lower[i] or close[i] < ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend changes to up
            if close[i] > upper[i] or close[i] > ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above upper channel with uptrend
                if close[i] > upper[i] and close[i] > ema_20_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower channel with downtrend
                elif close[i] < lower[i] and close[i] < ema_20_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals