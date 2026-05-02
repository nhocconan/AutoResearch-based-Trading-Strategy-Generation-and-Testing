#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR-based volume confirmation and 1w trend filter
# Uses 6h primary timeframe with 1d HTF for ATR-based volume spike detection (ATR ratio > 1.5)
# and 1w EMA50 for trend alignment to reduce whipsaw and capture institutional moves.
# Donchian breakouts capture momentum; volume confirmation filters false breakouts.
# Weekly trend filter ensures trades align with higher-timeframe momentum.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in both bull and bear markets by following the 1w trend direction only.

name = "6h_Donchian20_1dATR_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d for volatility normalization
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR(30) on 1d for longer-term volatility
    atr_30_1d = tr.rolling(window=30, min_periods=30).mean().values
    
    # ATR ratio (short-term/long-term volatility) for volume spike detection
    atr_ratio = atr_14_1d / atr_30_1d
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Align 1w EMA to 6h timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 55  # max(50 for 1w EMA, 30 for ATR, 20 for Donchian/volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + ATR ratio > 1.2 (vol expansion) + 1w uptrend + volume spike
            if (close[i] > high_20[i] and 
                atr_ratio_aligned[i] > 1.2 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + ATR ratio > 1.2 (vol expansion) + 1w downtrend + volume spike
            elif (close[i] < low_20[i] and 
                  atr_ratio_aligned[i] > 1.2 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below Donchian middle (mean reversion) or ATR ratio < 0.8 (vol contraction)
            donchian_mid = (high_20[i] + low_20[i]) / 2
            if close[i] < donchian_mid or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above Donchian middle (mean reversion) or ATR ratio < 0.8 (vol contraction)
            donchian_mid = (high_20[i] + low_20[i]) / 2
            if close[i] > donchian_mid or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals