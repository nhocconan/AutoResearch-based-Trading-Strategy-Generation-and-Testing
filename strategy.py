#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume surge and ATR filter.
# Long when price breaks above 12h Donchian(20) high AND 1d volume surge AND ATR(12h) < 100-day percentile (low volatility).
# Short when price breaks below 12h Donchian(20) low AND 1d volume surge AND ATR(12h) < 100-day percentile.
# Uses 1d volume surge for momentum and ATR percentile to avoid high-volatility whipsaws.
# Designed for fewer trades (target: 15-25/year) to reduce fee drag and improve generalization.
# Works in both bull and bear markets by following 12h breakouts with volatility filter.
name = "12h_Donchian20_VolumeSurge_ATRPercentile"
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
    
    # Load 1d data for volume surge
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume surge: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 2.0
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d)
    
    # Load 12h data for Donchian and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # 12h ATR (14-period) and its 100-day percentile for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr_list = []
    for i in range(len(close_12h)):
        if i == 0:
            tr = high_12h[0] - low_12h[0]
        else:
            tr = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]), abs(low_12h[i] - close_12h[i-1]))
        tr_list.append(tr)
    tr_12h = np.array(tr_list)
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 100-period percentile of ATR (approximate with 100-day lookback)
    atr_percentile_100 = pd.Series(atr_14_12h).rolling(window=100, min_periods=20).quantile(0.2).values
    low_volatility = atr_14_12h < atr_percentile_100
    low_volatility_aligned = align_htf_to_ltf(prices, df_12h, low_volatility)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(vol_surge_1d_aligned[i]) or np.isnan(low_volatility_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian high, volume surge, low volatility
            long_condition = (close[i] > donchian_high_20_aligned[i]) and vol_surge_1d_aligned[i] and low_volatility_aligned[i]
            # Short condition: break below Donchian low, volume surge, low volatility
            short_condition = (close[i] < donchian_low_20_aligned[i]) and vol_surge_1d_aligned[i] and low_volatility_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or volatility increases
            if (close[i] < donchian_low_20_aligned[i]) or (~low_volatility_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or volatility increases
            if (close[i] > donchian_high_20_aligned[i]) or (~low_volatility_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals