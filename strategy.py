#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 12:
        return np.zeros(n)
    
    # Weekly EMA12 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_12_1w = close_1w_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_12_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_12_1w)
    
    # Load daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Load 12h data for entry timing (price action)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_12_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high with volume surge AND weekly uptrend AND low volatility
            if (close[i] > donch_high_aligned[i] and 
                volume[i] > 2.5 * vol_avg_20[i] and 
                close[i] > ema_12_1w_aligned[i] and
                atr_14_aligned[i] < np.median(atr_14_aligned[max(0, i-50):i+1])):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian low with volume surge AND weekly downtrend AND low volatility
            elif (close[i] < donch_low_aligned[i] and 
                  volume[i] > 2.5 * vol_avg_20[i] and 
                  close[i] < ema_12_1w_aligned[i] and
                  atr_14_aligned[i] < np.median(atr_14_aligned[max(0, i-50):i+1])):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price reverts to midpoint of Donchian channel
            if position == 1:
                donch_mid = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
                if close[i] < donch_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                donch_mid = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
                if close[i] > donch_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian20_WeeklyEMA12_VolumeFilter"
timeframe = "12h"
leverage = 1.0