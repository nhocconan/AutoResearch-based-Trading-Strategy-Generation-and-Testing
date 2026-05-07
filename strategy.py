#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# Long when Lips > Teeth > Jaw AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period average
# Exit when Alligator lines cross in opposite direction or volume filter fails
# Designed for 1d timeframe with low trade frequency (target: 10-25/year) to minimize fee drag.
# Uses 1w EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "1d_WilliamsAlligator_1wEMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def smma(values, period):
    """Smoothed Moving Average (SMMA)"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 AND volume filter
            long_cond = (lips[i] > teeth[i]) and (teeth[i] > jaw[i]) and (close[i] > ema50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 AND volume filter
            short_cond = (lips[i] < teeth[i]) and (teeth[i] < jaw[i]) and (close[i] < ema50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alignment (Lips < Teeth < Jaw) OR volume filter fails
            if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment (Lips > Teeth > Jaw) OR volume filter fails
            if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals