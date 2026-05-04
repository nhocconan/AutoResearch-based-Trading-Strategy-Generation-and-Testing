#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>2.0x 20 EMA volume)
# Uses 4h Donchian channel breakouts for structure - captures strong momentum at key support/resistance
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>2.0x average volume) - tight to reduce trades to 19-50/year target
# Discrete sizing 0.30 balances profit potential with drawdown control
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets (continuation at upper band) and bear markets (continuation at lower band)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v2"
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
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period) from prior completed 4h bar
    # We need to get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe and shift by 1 bar to avoid look-ahead (use completed bar)
    upper_4h_shifted = np.roll(upper_4h, 1)
    upper_4h_shifted[0] = np.nan
    lower_4h_shifted = np.roll(lower_4h, 1)
    lower_4h_shifted[0] = np.nan
    
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h_shifted)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band AND price > 1d EMA50 AND volume spike
            if close[i] > upper_4h_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below lower Donchian band AND price < 1d EMA50 AND volume spike
            elif close[i] < lower_4h_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian band OR price crosses below 1d EMA50
            if close[i] < lower_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to upper Donchian band OR price crosses above 1d EMA50
            if close[i] > upper_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals