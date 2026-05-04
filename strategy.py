#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA(50) trend filter + volume spike confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trending vs ranging markets
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>2.0x 20 EMA volume) filters false breakouts in choppy markets
# Discrete sizing 0.25 minimizes fee churn while targeting 50-150 total trades over 4 years
# Works in bull markets (Alligator awake, trending up) and bear markets (Alligator awake, trending down)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_Balanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + current_price) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator awake (Lips > Teeth > Jaw) AND price > 1d EMA50 AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator awake (Lips < Teeth < Jaw) AND price < 1d EMA50 AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator sleeping (Jaw > Teeth > Lips) OR price crosses below 1d EMA50
            if jaw[i] > teeth[i] and teeth[i] > lips[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator sleeping (Jaw > Teeth > Lips) OR price crosses above 1d EMA50
            if jaw[i] > teeth[i] and teeth[i] > lips[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals