#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Uses Alligator (Jaw/Teeth/Lips) to identify trends and avoid whipsaws.
# Requires price outside Alligator mouth, aligned with 1w EMA50 trend, and volume confirmation.
# Designed for low-frequency trading (12-37 trades/year) to minimize fee drag.
# Works in both bull and bear markets by following weekly trend direction.

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
    
    # Calculate 50-period EMA on weekly close for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Alligator parameters (13, 8, 5) with SMMA
    def smoothed_moving_average(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smoothed_moving_average(close, 13)  # Blue line (13-period)
    teeth = smoothed_moving_average(close, 8)   # Red line (8-period)
    lips = smoothed_moving_average(close, 5)    # Green line (5-period)
    
    # Volume spike filter (50-period on 12h)
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > 1.5 * vol_ma50
    
    # Session filter: 00-24 UTC (always active for 12h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = ((hours >= 0) & (hours <= 23))  # Always true, but kept for structure
    
    # Align indicators to 12-hour timeframe
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma50[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above Alligator (lips > teeth > jaw) + volume spike + weekly uptrend
            if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and 
                vol_spike[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below Alligator (jaw > teeth > lips) + volume spike + weekly downtrend
            elif (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and 
                  vol_spike[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price re-enters Alligator mouth (lips crosses jaw or teeth)
            if position == 1:
                if lips_aligned[i] <= jaw_aligned[i]:  # Lips cross below jaw
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips_aligned[i] >= jaw_aligned[i]:  # Lips cross above jaw
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0