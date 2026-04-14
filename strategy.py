#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Channel Breakout with 1d ADX Trend Filter and Volume Spike
# Uses Donchian Channel (20-period high/low) for breakout entries
# 1d ADX (14) provides trend strength filter to avoid choppy markets
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d ADX data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX(14) on daily data
    plus_dm = np.diff(df_1d['high'], prepend=df_1d['high'].iloc[0])
    minus_dm = np.diff(df_1d['low'], prepend=df_1d['low'].iloc[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum(
        np.maximum(df_1d['high'] - df_1d['low'], np.abs(df_1d['high'] - df_1d['close'].shift(1))),
        np.abs(df_1d['low'] - df_1d['close'].shift(1))
    )
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_14 = dx.rolling(window=14, min_periods=14).mean()
    adx_14_values = adx_14.values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_values)
    
    # Donchian Channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for ADX and Donchian calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_14_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_14_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter and strong trend
            if price > highest_20[i] and vol > 1.5 * avg_vol[i] and strong_trend:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter and strong trend
            elif price < lowest_20[i] and vol > 1.5 * avg_vol[i] and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low (stop and reverse)
            if price < lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high (stop and reverse)
            if price > highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0