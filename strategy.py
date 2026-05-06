#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and ATR volatility filter
# Uses Donchian channels for structure on 12h timeframe, 1d EMA50 for strong trend alignment (reduces whipsaw)
# ATR(14) > 20-bar average ATR filters for sufficient volatility to avoid choppy markets
# Discrete sizing 0.25 to limit fee drag; target 50-150 trades over 4 years (12-37/year)
# Proven pattern: price channel breakouts with volume/volatility confirmation work on BTC/ETH in both bull/bear

name = "12h_Donchian20_1dEMA50_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for volatility filter on 12h
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    avg_atr_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > (1.2 * avg_atr_20)  # Require above-average volatility
    
    # Align HTF indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_12h, volatility_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volatility_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian AND uptrend (price > EMA50) AND sufficient volatility
            if close[i] > high_20_aligned[i] and close[i] > ema50_1d_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower Donchian AND downtrend (price < EMA50) AND sufficient volatility
            elif close[i] < low_20_aligned[i] and close[i] < ema50_1d_aligned[i] and volatility_filter_aligned[i]:
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