#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Alligator's Jaw/Teeth/Lips (SMAs 13/8/5) to detect trends in both bull/bear markets.
# Entry when Lips cross above Teeth (bullish) or below Teeth (bearish) with 1d EMA50 trend alignment.
# Volume filter confirms momentum. Target: 12-37 trades/year (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smoothed_moving_average(data, period):
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        # Apply Wilder smoothing: first value is SMA, subsequent: (prev*(period-1) + current)/period
        smoothed = np.full_like(data, np.nan, dtype=float)
        if len(sma) >= period:
            smoothed[period-1] = sma[period-1]
            for i in range(period, len(data)):
                if not np.isnan(sma[i]):
                    smoothed[i] = (smoothed[i-1] * (period-1) + sma[i]) / period
                else:
                    smoothed[i] = smoothed[i-1]
        return smoothed
    
    jaw = smoothed_moving_average(close, 13)
    teeth = smoothed_moving_average(close, 8)
    lips = smoothed_moving_average(close, 5)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish: Lips cross above Teeth with 1d uptrend and volume confirmation
            if (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * np.nanmedian(volume[max(0, i-20):i+1])):
                signals[i] = 0.25
                position = 1
            # Bearish: Lips cross below Teeth with 1d downtrend and volume confirmation
            elif (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * np.nanmedian(volume[max(0, i-20):i+1])):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Lips cross back to Jaw (trend weakness) or opposite signal
            if position == 1:
                if lips[i] < jaw[i]:  # Lips below Jaw = trend weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > jaw[i]:  # Lips above Jaw = trend weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dTrend_Volume_Session"
timeframe = "12h"
leverage = 1.0