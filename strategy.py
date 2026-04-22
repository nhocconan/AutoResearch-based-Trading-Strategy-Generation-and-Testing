#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour 123 Reversal pattern with 1d EMA34 trend filter and volume spike
    # Works in bull/bear via trend filter: only take long in uptrend, short in downtrend.
    # 123 pattern captures short-term reversals at swing points; EMA34 filters trend; volume confirms.
    # Targets ~20-30 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 123 pattern components on 4h data
    # Need swing highs and lows - using 5-period lookback/forward for simplicity
    # Swing high: high[i] is highest in window [i-2, i+2]
    # Swing low: low[i] is lowest in window [i-2, i+2]
    window = 5
    half_window = window // 2
    
    # Initialize arrays for swing points
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    # Calculate swing points (avoiding look-ahead by using only past data)
    for i in range(half_window, n - half_window):
        # Check if current high is the highest in the window
        if high[i] == np.max(high[i-half_window:i+half_window+1]):
            swing_high[i] = high[i]
        # Check if current low is the lowest in the window
        if low[i] == np.min(low[i-half_window:i+half_window+1]):
            swing_low[i] = low[i]
    
    # For the 123 pattern, we need to identify:
    # Point 1: swing high (for short) or swing low (for long)
    # Point 2: pullback/pullup
    # Point 3: failure to make new high/low
    
    # Track recent swing points for pattern detection
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    # Forward fill swing points (using only past information)
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_low = swing_low[i]
        last_swing_high[i] = last_high
        last_swing_low[i] = last_low
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20  # Require 1.8x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long 123 pattern: 
            # 1. Make swing low (point 1)
            # 2. Pullback to point 2 (higher low)
            # 3. Failed breakdown below point 1 (point 3) and close above point 2
            if (not np.isnan(last_swing_low[i]) and 
                low[i] <= last_swing_low[i] * 1.005 and  # Near swing low (within 0.5%)
                i >= 3 and low[i-1] > low[i-3] and  # Pullback: higher low
                close[i] > close[i-1] and  # Close up
                vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):  # Uptrend filter
                signals[i] = 0.25
                position = 1
            # Short 123 pattern:
            # 1. Make swing high (point 1)
            # 2. Pullback to point 2 (lower high)
            # 3. Failed breakout above point 1 (point 3) and close below point 2
            elif (not np.isnan(last_swing_high[i]) and 
                  high[i] >= last_swing_high[i] * 0.995 and  # Near swing high (within 0.5%)
                  i >= 3 and high[i-1] < high[i-3] and  # Pullback: lower high
                  close[i] < close[i-1] and  # Close down
                  vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):  # Downtrend filter
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: 
            # For long: price breaks below the swing low that started the pattern
            # For short: price breaks above the swing high that started the pattern
            # Or trend reversal vs 1d EMA34
            if position == 1:
                if (not np.isnan(last_swing_low[i]) and close[i] < last_swing_low[i] * 0.995) or \
                   close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (not np.isnan(last_swing_high[i]) and close[i] > last_swing_high[i] * 1.005) or \
                   close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_123_Pattern_1dEMA34_Volume_Spike_Session_v1"
timeframe = "4h"
leverage = 1.0