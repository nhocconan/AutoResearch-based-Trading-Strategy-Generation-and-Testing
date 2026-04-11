#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(13) for Elder Ray
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Smooth the power signals with EMA(13) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    
    # 6h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(150, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.3x average)
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Long signal: Bull Power > 0 and increasing (momentum)
        long_signal = volume_confirmed and (bull_power_6h[i] > 0) and (bull_power_6h[i] > bull_power_6h[i-1])
        
        # Short signal: Bear Power < 0 and decreasing (momentum)
        short_signal = volume_confirmed and (bear_power_6h[i] < 0) and (bear_power_6h[i] < bear_power_6h[i-1])
        
        # Exit when power crosses zero (mean reversion)
        exit_long = position == 1 and bull_power_6h[i] <= 0
        exit_short = position == -1 and bear_power_6h[i] >= 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Elder Ray Power strategy on 6h timeframe using daily Bull/Bear Power.
# Elder Ray measures the power of bulls (High - EMA13) and bears (Low - EMA13).
# Goes long when Bull Power is positive and rising with volume confirmation.
# Goes short when Bear Power is negative and falling with volume confirmation.
# Exits when power crosses zero, indicating weakening momentum.
# Daily timeframe provides stable trend context, 6h provides timely entries.
# Works in both bull and bear markets by measuring actual bull/bear strength.
# Volume confirmation filters out weak moves. Target: 60-100 trades over 4 years.