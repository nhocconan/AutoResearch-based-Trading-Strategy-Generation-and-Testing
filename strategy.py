#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1w Fibonacci extension levels (127.2% and 161.8%) for trend continuation.
# Uses 1w trend filter (EMA50) to determine direction. In uptrend, buy pullbacks to 127.2% extension;
# in downtrend, sell rallies to 161.8% extension. Volume confirmation reduces false signals.
# Designed for low trade frequency (~20-30/year) to minimize fee drag on 12h timeframe.
# Works in bull/bear by aligning with higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for Fibonacci extension levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly swing points for Fibonacci extension
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Find recent swing high and low (50-period lookback)
    swing_high = np.full(len(high_1w), np.nan)
    swing_low = np.full(len(low_1w), np.nan)
    
    for i in range(50, len(high_1w)):
        swing_high[i] = np.max(high_1w[i-50:i])
        swing_low[i] = np.min(low_1w[i-50:i])
    
    # Calculate Fibonacci extension levels: 127.2% and 161.8%
    # In uptrend: extension above swing high
    # In downtrend: extension below swing low
    diff = swing_high - swing_low
    ext_127 = swing_high + 0.272 * diff  # 127.2% extension
    ext_161 = swing_low - 0.618 * diff   # 161.8% extension
    
    # Align Fibonacci levels to 12h timeframe
    ext_127_aligned = align_htf_to_ltf(prices, df_1w, ext_127)
    ext_161_aligned = align_htf_to_ltf(prices, df_1w, ext_161)
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h ATR(20) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 12h volume ratio (current / 50-period average)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.where(vol_ma_50 == 0, 1, vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ext_127_aligned[i]) or np.isnan(ext_161_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_20[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ext127 = ext_127_aligned[i]
        ext161 = ext_161_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        atr = atr_20[i]
        vol_ratio_12h = vol_ratio[i]
        
        # Determine trend from daily EMA50
        uptrend = price > ema_trend
        downtrend = price < ema_trend
        
        # Volatility filter: avoid extreme volatility
        atr_ma_50 = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values[i]
        vol_filter = (atr < 2.5 * atr_ma_50)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_12h > 1.3)
        
        if position == 0:
            # In uptrend: look for long near 127.2% extension (pullback)
            if uptrend and vol_filter:
                if price <= ext127 * 1.005:  # Near 127.2% extension with small buffer
                    signals[i] = 0.25
                    position = 1
            # In downtrend: look for short near 161.8% extension (bounce)
            elif downtrend and vol_filter:
                if price >= ext161 * 0.995:  # Near 161.8% extension with small buffer
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches 161.8% extension or trend reverses
            if price >= ext161 * 0.995 or price < ema_trend or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches 127.2% extension or trend reverses
            if price <= ext127 * 1.005 or price > ema_trend or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_FibExtension_TrendPullback_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0