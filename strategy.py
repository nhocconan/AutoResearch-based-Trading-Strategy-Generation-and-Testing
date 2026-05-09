#!/usr/bin/env python3
# Hypothesis: 1h Fibonacci retracement levels derived from 4h swing points with volume confirmation
# Uses 4h swing highs/lows to identify key Fibonacci levels (38.2%, 61.8%) and enters on retracements
# Only takes longs when price retraces to 61.8% fib level AND 4h trend is up AND volume spike
# Only takes shorts when price retraces to 38.2% fib level AND 4h trend is down AND volume spike
# Exits when price reaches the opposite fib level or 4h trend reverses
# Target: 15-35 trades per year with position size 0.20 for controlled risk
# Uses session filter (08-20 UTC) to avoid low-liquidity periods

name = "1h_Fibonacci_Retracement_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for swing points and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h swing high and swing low (using 3-bar lookback)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Find swing points: swing high when current high > previous 2 and next 2 highs
    # Since we can't look ahead, we use previous bar's swing point
    swing_high = np.full_like(high_4h, np.nan)
    swing_low = np.full_like(low_4h, np.nan)
    
    for i in range(2, len(high_4h) - 2):
        if (high_4h[i] > high_4h[i-1] and high_4h[i] > high_4h[i-2] and
            high_4h[i] > high_4h[i+1] and high_4h[i] > high_4h[i+2]):
            swing_high[i] = high_4h[i]
        if (low_4h[i] < low_4h[i-1] and low_4h[i] < low_4h[i-2] and
            low_4h[i] < low_4h[i+1] and low_4h[i] < low_4h[i+2]):
            swing_low[i] = low_4h[i]
    
    # Forward fill swing points to get the most recent swing
    swing_high_series = pd.Series(swing_high)
    swing_low_series = pd.Series(swing_low)
    swing_high_ffill = swing_high_series.ffill().values
    swing_low_ffill = swing_low_series.ffill().values
    
    # Calculate Fibonacci levels: 0.382 and 0.618 retracement
    diff = swing_high_ffill - swing_low_ffill
    fib_382 = swing_low_ffill + 0.382 * diff
    fib_618 = swing_low_ffill + 0.618 * diff
    
    # Align Fibonacci levels to 1h timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_4h, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_4h, fib_618)
    
    # 4h trend: 20-period EMA slope
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_prev = np.roll(ema_20_4h, 1)
    ema_20_prev[0] = ema_20_4h[0]
    ema_rising = ema_20_4h > ema_20_prev
    ema_falling = ema_20_4h < ema_20_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_falling)
    
    # Volume spike: current volume > 2.0x 24-period average volume (more restrictive)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(fib_382_aligned[i]) or np.isnan(fib_618_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price at 61.8% fib + 4h trend up + volume spike
            if (abs(close[i] - fib_618_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of fib level
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price at 38.2% fib + 4h trend down + volume spike
            elif (abs(close[i] - fib_382_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of fib level
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price reaches 38.2% fib OR trend turns down
            if (close[i] <= fib_382_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price reaches 61.8% fib OR trend turns up
            if (close[i] >= fib_618_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals