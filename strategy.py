#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout_With_Volume_and_Trend_v1
Hypothesis: Use 12h timeframe to reduce trade frequency and avoid fee drag. Breakout above Keltner upper band (EMA + ATR multiplier) with volume confirmation and trend alignment (price > weekly EMA50) for longs. Opposite for shorts. Keltner channels adapt to volatility, providing dynamic support/resistance. Weekly trend filter ensures we only trade in direction of higher timeframe momentum, reducing false signals in chop. Volume confirms institutional participation. Designed for 15-30 trades/year to minimize fee drag while capturing high-probability moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA50 for trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h EMA20 for Keltner middle line
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 12h ATR(10) for Keltner width
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = np.maximum(high_12h, np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2 = np.maximum(low_12h, np.roll(close_12h, 1))
    tr2[0] = high_12h[0] - low_12h[0]
    tr = np.maximum(tr1 - tr2, high_12h - low_12h)
    tr = np.where(tr < 0, high_12h - low_12h, tr)
    atr_10_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_10_12h)
    
    # Keltner bands: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema_20_12h_aligned + 2 * atr_10_12h_aligned
    keltner_lower = ema_20_12h_aligned - 2 * atr_10_12h_aligned
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_trend = ema_50_1w_aligned[i]
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > upper band with volume spike and above weekly EMA50
            if price > upper and vol_spike and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price < lower band with volume spike and below weekly EMA50
            elif price < lower and vol_spike and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < lower band or below weekly EMA50
            if price < lower or price < weekly_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > upper band or above weekly EMA50
            if price > upper or price > weekly_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Keltner_Channel_Breakout_With_Volume_and_Trend_v1"
timeframe = "12h"
leverage = 1.0