#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA50 for trend filter (requires 50 weeks of data)
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 1-day data for 12-hour pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Calculate pivot points from previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_range = np.roll(daily_range, 1)
    
    # Set first day values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_range[0] = np.nan
    
    # Calculate pivot point and support/resistance levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = pp + (prev_range * 1.1 / 12)
    s1 = pp - (prev_range * 1.1 / 12)
    
    # Volume spike filter (10-period on 12h)
    vol_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_spike = volume > 2.0 * vol_ma10
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 12-hour timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma10[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above 1w EMA50 + breaks above R1 + volume spike
            if (close[i] > ema50_1w_aligned[i] and 
                close[i] > r1_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below 1w EMA50 + breaks below S1 + volume spike
            elif (close[i] < ema50_1w_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite level or trend changes
            if position == 1:
                if (close[i] < s1_aligned[i] or 
                    close[i] < ema50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > r1_aligned[i] or 
                    close[i] > ema50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Pivot_EMA50_Trend_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0