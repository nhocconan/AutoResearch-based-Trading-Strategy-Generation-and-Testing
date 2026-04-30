#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction, 1h Williams %R for entry timing, and session filter (08-20 UTC) to reduce noise.
# Long when 4h Supertrend is bullish, 1h Williams %R crosses above -80 from below, and in session.
# Short when 4h Supertrend is bearish, 1h Williams %R crosses below -20 from above, and in session.
# Exit on opposite Williams %R cross (-20 for long, -80 for short) or Supertrend flip.
# Uses proven Supertrend trend filter with Williams %R momentum entries to target 15-35 trades/year.
# Session filter reduces choppy off-hour trades. Timeframe: 1h, HTF: 4h for trend.

name = "1h_Supertrend4h_WilliamsR_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0] if len(close_4h) > 0 else 0), np.abs(low_4h[0] - close_4h[0] if len(close_4h) > 0 else 0)])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high_4h + low_4h) / 2
    upper_band = hl_avg + (3.0 * atr_10)
    lower_band = hl_avg - (3.0 * atr_10)
    
    # Supertrend calculation
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align 4h Supertrend direction to 1h timeframe (completed 4h bar only)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Williams %R on 1h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Williams %R and Supertrend
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_supertrend_dir = supertrend_dir_aligned[i]
        prev_wr = williams_r[i-1] if i > 0 else -50
        
        if position == 0:  # Flat - look for new entries
            # Long: Supertrend bullish, Williams %R crosses above -80 from below
            if (curr_supertrend_dir == 1 and 
                curr_wr > -80 and 
                prev_wr <= -80):
                signals[i] = 0.20
                position = 1
            # Short: Supertrend bearish, Williams %R crosses below -20 from above
            elif (curr_supertrend_dir == -1 and 
                  curr_wr < -20 and 
                  prev_wr >= -20):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -20 from above (overbought) OR Supertrend turns bearish
            if (curr_wr < -20 and prev_wr >= -20) or curr_supertrend_dir == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -80 from below (oversold) OR Supertrend turns bullish
            if (curr_wr > -80 and prev_wr <= -80) or curr_supertrend_dir == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals