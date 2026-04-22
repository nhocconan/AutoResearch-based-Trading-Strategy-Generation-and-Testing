#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6-hour 123 Reversal Pattern with 1-day trend filter and volume confirmation
    # 123 Reversal: Price makes new high/low (1), pulls back (2), then breaks pullback extreme (3)
    # In trending markets: strong continuation after pullback in direction of trend
    # 1-day EMA89 filters trend: only take longs in uptrend, shorts in downtrend
    # Volume spike confirms breakout of pullback
    # Targets ~20-30 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for 123 pattern
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA89 on 1d close for trend filter
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMA89 to 6h timeframe
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # Calculate 10-period high/low for pullback identification on 6h
    high_10 = pd.Series(high_6h).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_6h).rolling(window=10, min_periods=10).min().values
    
    # Align 6h indicators
    high_10_aligned = align_htf_to_ltf(prices, df_6h, high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_6h, low_10)
    
    # Volume spike filter (15-period on 6h)
    vol_ma15 = pd.Series(volume_6h).rolling(window=15, min_periods=15).mean().values
    vol_spike = volume_6h > 1.8 * vol_ma15  # Require 1.8x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    pullback_high = np.full(n, np.nan)
    pullback_low = np.full(n, np.nan)
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(high_10_aligned[i]) or np.isnan(low_10_aligned[i]) or
            np.isnan(ema89_1d_aligned[i]) or np.isnan(vol_ma15[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long 123: New high (1), pullback (2), break above pullback high (3)
            if (high_6h[i-1] == high_10_aligned[i-1] and  # New 10-period high
                high[i] > pullback_high[i-1] and           # Break above pullback high
                close[i] > ema89_1d_aligned[i] and         # Above 1d EMA89 (uptrend)
                vol_spike[i]):                             # Volume confirmation
                signals[i] = 0.25
                position = 1
                pullback_high[i] = high[i]  # Reset pullback tracking
            # Short 123: New low (1), pullback (2), break below pullback low (3)
            elif (low_6h[i-1] == low_10_aligned[i-1] and   # New 10-period low
                  low[i] < pullback_low[i-1] and           # Break below pullback low
                  close[i] < ema89_1d_aligned[i] and       # Below 1d EMA89 (downtrend)
                  vol_spike[i]):                           # Volume confirmation
                signals[i] = -0.25
                position = -1
                pullback_low[i] = low[i]   # Reset pullback tracking
            else:
                # Update pullback levels
                if position == 0:
                    if high_6h[i-1] == high_10_aligned[i-1]:  # After new high, track pullback low
                        pullback_high[i] = low[i]
                    elif low_6h[i-1] == low_10_aligned[i-1]:  # After new low, track pullback high
                        pullback_low[i] = high[i]
                    else:
                        pullback_high[i] = pullback_high[i-1]
                        pullback_low[i] = pullback_low[i-1]
                else:
                    pullback_high[i] = pullback_high[i-1]
                    pullback_low[i] = pullback_low[i-1]
        else:
            # Track pullback levels even when in position
            if position == 1:
                if high_6h[i-1] == high_10_aligned[i-1]:  # New high, reset pullback tracking
                    pullback_high[i] = low[i]
                else:
                    pullback_high[i] = pullback_high[i-1]
                pullback_low[i] = pullback_low[i-1]
                # Exit: Break below pullback low or trend reversal
                if low[i] < pullback_high[i] or close[i] < ema89_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if low_6h[i-1] == low_10_aligned[i-1]:  # New low, reset pullback tracking
                    pullback_low[i] = high[i]
                else:
                    pullback_low[i] = pullback_low[i-1]
                pullback_high[i] = pullback_high[i-1]
                # Exit: Break above pullback high or trend reversal
                if high[i] > pullback_low[i] or close[i] > ema89_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_123Reversal_1dEMA89_Trend_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0