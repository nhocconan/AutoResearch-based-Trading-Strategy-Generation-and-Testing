#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation
# Bullish fractal breakout (price above recent fractal high) when above weekly EMA200 (uptrend)
# Bearish fractal breakout (price below recent fractal low) when below weekly EMA200 (downtrend)
# Volume filter ensures breakouts have participation. Targets 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 (trend filter)
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Load daily data ONCE for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals (5-bar pattern)
    # Bullish fractal: low[n-2] < low[n-1] and low[n] < low[n-1] and low[n+1] < low[n-1] and low[n+2] < low[n-1]
    # Bearish fractal: high[n-2] > high[n-1] and high[n] > high[n-1] and high[n+1] > high[n-1] and high[n+2] > high[n-1]
    n_1d = len(high_1d)
    bullish_fractal = np.full(n_1d, np.nan)
    bearish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (low_1d[i-2] > low_1d[i-1] and low_1d[i] > low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1] and low_1d[i+2] > low_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Bearish fractal at i-1
        if (high_1d[i-2] < high_1d[i-1] and high_1d[i] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1] and high_1d[i+2] < high_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]   # Bullish fractal at i-1
    
    # Align fractals to 6h timeframe with 2-bar confirmation delay
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Calculate 6h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 6h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or \
           np.isnan(bearish_fractal_aligned[i]) or np.isnan(atr_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        bullish_fractal_level = bullish_fractal_aligned[i]
        bearish_fractal_level = bearish_fractal_aligned[i]
        weekly_ema = ema200_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price above bullish fractal support, above weekly EMA200 (uptrend), with volume
            if not np.isnan(bullish_fractal_level) and price > bullish_fractal_level and price > weekly_ema and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below bearish fractal resistance, below weekly EMA200 (downtrend), with volume
            elif not np.isnan(bearish_fractal_level) and price < bearish_fractal_level and price < weekly_ema and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below bullish fractal support
            if price <= entry_price - 2.0 * atr_6h[i] or (not np.isnan(bullish_fractal_level) and price < bullish_fractal_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above bearish fractal resistance
            if price >= entry_price + 2.0 * atr_6h[i] or (not np.isnan(bearish_fractal_level) and price > bearish_fractal_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_WeeklyEMA200_VolumeFilter"
timeframe = "6h"
leverage = 1.0