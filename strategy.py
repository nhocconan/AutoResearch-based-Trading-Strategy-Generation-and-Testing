#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-2-3 Pattern with 1w Trend Filter and Volume Spike
# - Uses 1-2-3 pattern formation (retest of swing points) on 12h timeframe
# - Confirmed by weekly trend direction to avoid counter-trend trades
# - Volume spike validates breakout strength
# - Works in bull/bear markets by using 1w trend filter
# - Target: 12-37 trades/year to minimize fee drag on 12h timeframe

name = "12h_123Pattern_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1-2-3 pattern points on 12h timeframe
    # Point 1: swing high/low
    # Point 2: retracement
    # Point 3: test of point 1 level
    lookback = 5
    
    # Find swing highs and lows
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(lookback, n - lookback):
        # Swing high: higher than lookback periods on both sides
        if high[i] == np.max(high[i-lookback:i+lookback+1]):
            swing_high[i] = high[i]
        # Swing low: lower than lookback periods on both sides
        if low[i] == np.min(low[i-lookback:i+lookback+1]):
            swing_low[i] = low[i]
    
    # For each point, find the most recent swing point
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_low = swing_low[i]
        last_swing_high[i] = last_high
        last_swing_low[i] = last_low
    
    # Calculate potential 1-2-3 pattern completion
    # Bullish pattern: swing low (1) -> retracement to higher low (2) -> test of swing low (3)
    # Bearish pattern: swing high (1) -> retracement to lower high (2) -> test of swing high (3)
    
    # Bullish 1-2-3: price tests and holds above prior swing low
    bullish_setup = (
        (last_swing_low > 0) & 
        (close > last_swing_low * 0.995) &  # within 0.5% of swing low (point 3)
        (low <= last_swing_low * 1.01) &    # touched or went below swing low (point 1)
        (close > last_swing_low)            # now back above swing low
    )
    
    # Bearish 1-2-3: price tests and holds below prior swing high
    bearish_setup = (
        (last_swing_high > 0) & 
        (close < last_swing_high * 1.005) &  # within 0.5% of swing high (point 3)
        (high >= last_swing_high * 0.99) &   # touched or went above swing high (point 1)
        (close < last_swing_high)            # now back below swing high
    )
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish 1-2-3 completion with 1w uptrend + volume spike
            long_cond = bullish_setup[i] and (ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]) and volume_spike[i]
            
            # Short: bearish 1-2-3 completion with 1w downtrend + volume spike
            short_cond = bearish_setup[i] and (ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below the swing low that formed the pattern
            if close[i] < last_swing_low[i] * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above the swing high that formed the pattern
            if close[i] > last_swing_high[i] * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals