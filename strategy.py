#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume confirmation
# Donchian(20) breakout captures strong momentum moves; 1w EMA21 filter ensures alignment with weekly trend
# Volume confirmation (>2x 20-bar EMA) reduces false breakouts
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag
# Works in bull markets via breakouts with trend, and in bear markets via short breakdowns against weekly downtrend

name = "1d_Donchian20_1wEMA21_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA21 trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA21 to daily timeframe (wait for completed 1w bar)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate Donchian(20) channels on daily data
    # Use rolling window with min_periods to avoid look-ahead
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(high_max_20[i]) or np.isnan(low_min_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper AND price > 1w EMA21 (bullish weekly trend) AND volume spike
            if close[i] > high_max_20[i] and close[i] > ema21_1w_aligned[i] and volume[i] > (2.0 * pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().iloc[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower AND price < 1w EMA21 (bearish weekly trend) AND volume spike
            elif close[i] < low_min_20[i] and close[i] < ema21_1w_aligned[i] and volume[i] > (2.0 * pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().iloc[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian upper OR below 1w EMA21
            if close[i] < high_max_20[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian lower OR above 1w EMA21
            if close[i] > low_min_20[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals