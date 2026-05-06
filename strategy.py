#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA200 trend filter and ATR volatility filter
# Uses Donchian channels for structure, 12h EMA200 for strong trend alignment (reduces bear market whipsaw)
# ATR(14) > 20-bar average ATR filters for sufficient volatility to avoid choppy markets
# Discrete sizing 0.25 to limit fee drag; target 75-200 trades over 4 years
# Proven pattern: price channel breakouts with volume/volatility confirmation work on BTC/ETH in both bull/bear

name = "4h_Donchian20_12hEMA200_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 20 or len(df_12h) < 200:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_12h = df_12h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA200 trend filter
    close_12h_series = pd.Series(close_12h)
    ema200_12h = close_12h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    avg_atr_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > (1.2 * avg_atr_20)  # Require above-average volatility
    
    # Align HTF indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_4h, volatility_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema200_12h_aligned[i]) or np.isnan(volatility_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian AND uptrend (price > EMA200) AND sufficient volatility
            if close[i] > high_20_aligned[i] and close[i] > ema200_12h_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower Donchian AND downtrend (price < EMA200) AND sufficient volatility
            elif close[i] < low_20_aligned[i] and close[i] < ema200_12h_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests lower Donchian from above (trend reversal)
            if close[i] <= low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests upper Donchian from below (trend reversal)
            if close[i] >= high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals