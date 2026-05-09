#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h timeframe for trend context and 1d for volatility context.
# Uses 12h EMA50 for trend direction and 1d ATR-based volatility filter to avoid choppy markets.
# Entry: price breaks above/below 4h Donchian(20) with trend alignment and volatility filter.
# Exit: reverse signal or volatility spike (avoid whipsaw).
# Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull/bear markets.
name = "4h_Donchian20_12hEMA50_1dATRFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_ma_50 = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14_1d / atr_ma_50
    atr_ratio_4h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(atr_ratio_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extreme volatility (ATR ratio > 2.0) and dead zones (ATR ratio < 0.5)
        vol_ok = 0.5 <= atr_ratio_4h[i] <= 2.0
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above Donchian upper band, uptrend (price > EMA50), volatility OK
            if (close[i] > high_max_20[i] and 
                close[i] > ema_50_4h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, downtrend (price < EMA50), volatility OK
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema_50_4h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower band or trend reversal
            if close[i] < low_min_20[i] or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper band or trend reversal
            if close[i] > high_max_20[i] or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals