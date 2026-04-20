#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_PriceAction_VolatilityBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate volatility breakout levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Average True Range for volatility
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Upper and lower breakout bands: mean close ± 1.5 * ATR
    mean_close = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    upper_band = mean_close + 1.5 * atr_1d
    lower_band = mean_close - 1.5 * atr_1d
    
    # Align to 6h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    mean_close_aligned = align_htf_to_ltf(prices, df_1d, mean_close)
    
    # === 6h: Price action and volume confirmation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_condition = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        mean = mean_close_aligned[i]
        vol_ok = vol_condition[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(mean) or np.isnan(vol_ok)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band with volume
            if current_close > upper and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: price breaks below lower band with volume
            elif current_close < lower and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price reverts to mean or stop loss
            if current_close <= mean or current_close < entry_price - 2.0 * (upper - lower):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reverts to mean or stop loss
            if current_close >= mean or current_close > entry_price + 2.0 * (upper - lower):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals