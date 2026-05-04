#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Donchian(20) breakout captures strong trends, volume spike confirms institutional participation,
# choppiness regime filter (CHOP > 61.8) avoids whipsaw in ranging markets.
# Works in both bull and bear markets by only taking breakouts aligned with the 1d EMA50 trend.

name = "12h_Donchian20_1dVolumeSpike_ChopFilter_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume MA and choppiness - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Calculate 1d choppiness index (CHOP) - high when ranging, low when trending
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(n)
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index 0
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND chop > 61.8 (ranging) AND price > 1d EMA50
            if (close[i] > donchian_high[i] and 
                volume_spike_aligned[i] and 
                chop_1d_aligned[i] > 61.8 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND chop > 61.8 (ranging) AND price < 1d EMA50
            elif (close[i] < donchian_low[i] and 
                  volume_spike_aligned[i] and 
                  chop_1d_aligned[i] > 61.8 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR chop < 38.2 (strong trend - consider reversal)
            if (close[i] <= donchian_high[i] and close[i] >= donchian_low[i]) or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR chop < 38.2 (strong trend - consider reversal)
            if (close[i] <= donchian_high[i] and close[i] >= donchian_low[i]) or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals