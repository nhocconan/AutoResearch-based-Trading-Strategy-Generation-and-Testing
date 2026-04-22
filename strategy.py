#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# The Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend presence when lines are aligned and separated.
# In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend), we trade pullbacks to the Teeth line.
# Uses 1d EMA50 for higher timeframe trend filter and volume spike for confirmation.
# Works in both bull and bear markets by following the higher timeframe trend.
# Uses discrete position sizing (0.25) to limit transaction costs.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator lines (Smoothed Moving Average)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA calculation: SMMA_t = (SMMA_{t-1} * (period-1) + close_t) / period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator alignment: JAW > TEETH > LIPS for uptrend, reverse for downtrend
            # Long: Bullish alignment + price > Teeth + above 1d EMA + volume spike
            if jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] > teeth[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < Teeth + below 1d EMA + volume spike
            elif jaw[i] < teeth[i] and teeth[i] < lips[i] and close[i] < teeth[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Teeth line in opposite direction
            if position == 1:
                # Exit long: Close below Teeth
                if close[i] < teeth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Close above Teeth
                if close[i] > teeth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_TeethPullback_1dEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0