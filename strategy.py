#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Volume Spike + 4h/1d Trend Alignment
# Uses volume spikes (>2x 20-period average) as entry signals with directional bias
# from 4h Supertrend and 1d EMA200 to avoid counter-trend trades.
# Volume spikes often precede strong moves in both bull and bear markets.
# Trend filters ensure we trade in the direction of higher timeframe momentum.
# Target: 60-150 total trades over 4 years (15-37/year) with disciplined entries.
# Volume confirmation reduces false breakouts; trend alignment improves win rate.
name = "1h_VolumeSpike_4dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Supertrend for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate ATR for Supertrend
    atr_period = 10
    tr1 = pd.Series(df_4h['high']).diff().abs()
    tr2 = (pd.Series(df_4h['high']) - pd.Series(df_4h['low'].shift())).abs()
    tr3 = (pd.Series(df_4h['low']) - pd.Series(df_4h['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    
    # Supertrend calculation
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    multiplier = 3.0
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(df_4h['close'], np.nan, dtype=float)
    direction = np.full_like(df_4h['close'], np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(df_4h)):
        if i == 0:
            supertrend[i] = upperband.iloc[i]
            direction[i] = 1
        else:
            if close_4h := df_4h['close'].iloc[i] > supertrend[i-1]:
                supertrend[i] = max(upperband.iloc[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(lowerband.iloc[i], supertrend[i-1])
                direction[i] = -1
    
    # Align Supertrend direction to 1h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1d EMA200 for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume spike detection: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: volume spike + 4h uptrend + price above 1d EMA200
            if (volume_spike[i] and 
                supertrend_dir_aligned[i] == 1 and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: volume spike + 4h downtrend + price below 1d EMA200
            elif (volume_spike[i] and 
                  supertrend_dir_aligned[i] == -1 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if 4h trend turns down OR price breaks below 1d EMA200
            if (supertrend_dir_aligned[i] == -1) or (close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if 4h trend turns up OR price breaks above 1d EMA200
            if (supertrend_dir_aligned[i] == 1) or (close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals