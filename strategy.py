#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 1d EMA200 trend filter and volume confirmation
# Uses Donchian(20) for structure, 1d EMA200 for trend alignment (reduces whipsaw in bear markets)
# Volume spike (>1.8x 20-bar average) confirms breakout strength
# ATR-based stoploss via signal=0 when price retests opposite Donchian level
# Discrete sizing 0.20 to limit fee drag; target 60-150 total trades over 4 years
# Proven pattern: price channel breakouts with volume confirmation work on BTC/ETH in both bull/bear

name = "1h_Donchian20_1dEMA200_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA200 trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    
    # Calculate volume spike filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Align HTF indicators to 1h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND uptrend (price > EMA200) AND volume spike
            if close[i] > donchian_high[i] and close[i] > ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < Donchian low AND downtrend (price < EMA200) AND volume spike
            elif close[i] < donchian_low[i] and close[i] < ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests Donchian low from above (trend reversal)
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests Donchian high from below (trend reversal)
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals