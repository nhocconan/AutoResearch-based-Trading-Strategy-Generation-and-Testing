#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout with 1d Volume Confirmation and ADX Trend Filter
# Takes long when price breaks above recent Williams fractal high with 1d volume spike and ADX > 25
# Takes short when price breaks below recent Williams fractal low with 1d volume spike and ADX > 25
# Exits when price crosses back below/above the fractal level or volume drops
# Williams Fractals identify potential turning points; combining with volume and trend filters
# targets strong momentum moves while avoiding choppy markets. Designed for 12h timeframe
# to capture multi-day trends with controlled trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on 12h data (5-bar window: 2 left, 2 right)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Bullish fractal: lowest low in center with higher lows on both sides
    bullish_fractal = np.full(len(high_12h), np.nan)
    # Bearish fractal: highest high in center with lower highs on both sides
    bearish_fractal = np.full(len(high_12h), np.nan)
    
    for i in range(2, len(high_12h) - 2):
        # Bullish fractal: low[i-2] > low[i] and low[i-1] > low[i] and low[i+1] > low[i] and low[i+2] > low[i]
        if (low[i-2] > low[i] and low[i-1] > low[i] and 
            low[i+1] > low[i] and low[i+2] > low[i]):
            bullish_fractal[i] = low[i]
        # Bearish fractal: high[i-2] < high[i] and high[i-1] < high[i] and high[i+1] < high[i] and high[i+2] < high[i]
        if (high[i-2] < high[i] and high[i-1] < high[i] and 
            high[i+1] < high[i] and high[i+2] < high[i]):
            bearish_fractal[i] = high[i]
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100  # for fractal and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: break above recent bullish fractal with volume spike and strong trend
            if (price > bullish_fractal_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                adx_aligned[i] > 25):                           # Strong trend
                position = 1
                signals[i] = position_size
            # Short setup: break below recent bearish fractal with volume spike and strong trend
            elif (price < bearish_fractal_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                  adx_aligned[i] > 25):                           # Strong trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below bullish fractal level or volume drops
            if price < bullish_fractal_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above bearish fractal level or volume drops
            if price > bearish_fractal_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0