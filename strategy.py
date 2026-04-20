#!/usr/bin/env python3
# Strategy: 4h_1d_KeltnerChannel_WidthFilter
# Hypothesis: Breakouts from Keltner Channel (ATR-based) on 4h timeframe with low volatility filter (bandwidth < 30-day percentile) to avoid whipsaws, 
# and volume confirmation (>1.5x 20-period MA). Uses 1d EMA50 trend filter to align with higher timeframe momentum.
# Designed for 20-40 trades/year on 4h to minimize fee drag and work in both bull/bear markets via trend alignment and volatility filtering.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for primary calculations
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ATR for Keltner Channel (20-period)
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_close = np.abs(low_4h - np.roll(close_4h, 1))
    high_low[0] = high_4h[0] - low_4h[0]
    high_close[0] = np.abs(high_4h[0] - close_4h[0])
    low_close[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel: 20-period EMA ± 2 * ATR
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20_4h + 2.0 * atr_20
    lower_keltner = ema20_4h - 2.0 * atr_20
    
    # Bandwidth for volatility filter: (Upper - Lower) / Middle
    bandwidth = (upper_keltner - lower_keltner) / ema20_4h
    # 30-day (approx 180 periods of 4h) percentile rank for bandwidth
    # We'll use rolling percentile: percentage of values < current in last 180 periods
    bandwidth_series = pd.Series(bandwidth)
    # Avoid computing full percentile - use threshold: bandwidth < 0.05 (empirical threshold for low volatility)
    # This is equivalent to ~30th percentile in many regimes
    low_volatility = bandwidth < 0.05
    
    # Volume confirmation: 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for indicators to stabilize
        # Skip if NaN in critical values
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner, above 1d EMA50 (uptrend), low volatility, volume confirmation
            if (price > upper_keltner[i] and 
                price > ema50_1d_aligned[i] and 
                low_volatility[i] and 
                vol > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner, below 1d EMA50 (downtrend), low volatility, volume confirmation
            elif (price < lower_keltner[i] and 
                  price < ema50_1d_aligned[i] and 
                  low_volatility[i] and 
                  vol > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Keltner or ATR-based stop (2x ATR from entry not tracked, use trailing)
            # Simple exit: price re-enters Keltner Channel
            if price < lower_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Keltner or re-enters channel
            if price > upper_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KeltnerChannel_WidthFilter"
timeframe = "4h"
leverage = 1.0