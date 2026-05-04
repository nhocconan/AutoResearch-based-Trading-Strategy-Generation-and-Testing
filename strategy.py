#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR regime filter + volume confirmation
# Uses 1d ATR to define volatility regime: high ATR = trending (favor breakouts), low ATR = range (avoid breakouts)
# Combines with 4h Donchian breakout and 4h volume > 1.5x 20-period EMA for confirmation
# Designed for 4h timeframe targeting 20-50 trades/year with discrete sizing (0.25)
# Works in bull markets (breakouts during high volatility) and bear markets (breakouts during high volatility spikes)
# ATR regime filter prevents false breakouts in low volatility choppy periods
# Volume confirmation ensures institutional participation behind breakouts

name = "4h_Donchian20_1dATR_Regime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR as EMA of TR
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR regime: high volatility (trending) when ATR > 20-period EMA of ATR
    atr_ema_20 = pd.Series(atr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_regime = atr_1d > atr_ema_20  # True = high volatility/trending regime
    
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_channel = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Get 4h data for volume EMA
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ema_20 = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_regime_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + high volatility regime
            if (close[i] > upper_aligned[i] and volume_confirmed and atr_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + high volatility regime
            elif (close[i] < lower_aligned[i] and volume_confirmed and atr_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian OR volatility regime shifts to low
            if close[i] < lower_aligned[i] or not atr_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Donchian OR volatility regime shifts to low
            if close[i] > upper_aligned[i] or not atr_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals