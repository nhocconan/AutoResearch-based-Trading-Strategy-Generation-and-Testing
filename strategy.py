#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout with 4h trend filter and 1d volatility regime filter
# Uses 4h Donchian(20) for trend direction, 1h for entry timing, 1d ATR ratio for volatility filter
# Session filter 08-20 UTC to avoid low-volume periods
# Position size: 0.20 (discrete) to control drawdown
# Target: 15-30 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h
    upper_20_1h = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_1h = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # Get 1d HTF data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and MA(50) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / (atr_ma_50_1d + 1e-10)
    
    # Align 1d ATR ratio to 1h
    atr_ratio_1h = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    signals = np.zeros(n)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1h[i]) or np.isnan(lower_20_1h[i]) or 
            np.isnan(atr_ratio_1h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. Volatility regime: ATR ratio > 0.8 (avoid extremely low volatility)
        # 3. Volume confirmation: volume > 1.2x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 20 else np.mean(volume[:i+1])
        volume_ratio = volume[i] / (vol_ma_20 + 1e-10)
        
        if (close[i] > upper_20_1h[i] and
            atr_ratio_1h[i] > 0.8 and
            volume_ratio > 1.2):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 1h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. Volatility regime: ATR ratio > 0.8
        # 3. Volume confirmation: volume > 1.2x 20-period average
        elif (close[i] < lower_20_1h[i] and
              atr_ratio_1h[i] > 0.8 and
              volume_ratio > 1.2):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_Donchian20_1d_ATRRegime_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0