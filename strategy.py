#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout + Weekly Trend + Volume Spike
# Hypothesis: 1d breakouts in direction of weekly trend with volume capture strong moves
# while avoiding false breakouts. Works in bull (breakouts continue) and bear (breakouts reverse quickly for shorts).
# Target: 8-20 trades/year (32-80 total over 4 years) for 1d timeframe.

name = "1d_donchian_breakout_weekly_trend_volume_v1"
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
    
    # Weekly EMA(20) for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Donchian(20) channels on 1d
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_weekly_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or weekly trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or weekly trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Strong uptrend on weekly
                if close[i] > highest_high[i] and close[i] > ema_20_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Strong downtrend on weekly
                elif close[i] < lowest_low[i] and close[i] < ema_20_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals