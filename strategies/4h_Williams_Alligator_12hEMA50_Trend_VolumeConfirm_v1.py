#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams Alligator with 12h EMA50 trend filter and volume confirmation
    # Williams Alligator (Jaw/Teeth/Lips) identifies trend strength and direction
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # When Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
    # Entries on crossover with volume confirmation to avoid whipsaws
    # Target: 20-40 trades/year with controlled risk
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter (higher timeframe)
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams Alligator components (using SMMA - smoothed moving average)
    # SMMA is similar to EMA but with different smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: (prev*(period-1) + current) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (13-period)
    teeth = smma(close, 8)  # Teeth (8-period)
    lips = smma(close, 5)   # Lips (5-period)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after Alligator warmup
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume spike + price above 12h EMA50
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume spike + price below 12h EMA50
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross (trend weakening) or trend reversal vs 12h EMA
            if position == 1:
                if lips[i] < teeth[i] or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i] or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0