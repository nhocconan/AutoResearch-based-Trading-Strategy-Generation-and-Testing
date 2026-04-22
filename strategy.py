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
    
    # Load 12h data (HTF) and 1d data (for context) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.inf  # first bar has no previous close
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe
    donch_high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20_12h)
    donch_low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_12h_aligned[i]) or np.isnan(donch_low_20_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high with volume AND above 12h EMA50 (uptrend)
            # AND volatility is not extreme (ATR < 2 * ATR mean)
            atr_mean = np.nanmean(atr_1d_aligned[max(0, i-50):i+1])
            if (close[i] > donch_high_20_12h_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_12h_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and
                atr_1d_aligned[i] < 2 * atr_mean):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian low with volume AND below 12h EMA50 (downtrend)
            # AND volatility is not extreme
            elif (close[i] < donch_low_20_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_12h_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  atr_1d_aligned[i] < 2 * atr_mean):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite 12h Donchian level
            if position == 1:
                if close[i] < donch_low_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_high_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0