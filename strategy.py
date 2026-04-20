#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KC_Breakout_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Keltner Channel (20, 1.5) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA20 of close
    close_series = pd.Series(close_1d)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR20
    tr_series = pd.Series(tr)
    atr20 = tr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    kc_upper = ema20 + 1.5 * atr20
    kc_lower = ema20 - 1.5 * atr20
    
    # Align to 12h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
    
    # === 12h Volume Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        kc_upper_val = kc_upper_aligned[i]
        kc_lower_val = kc_lower_aligned[i]
        ema20_val = ema20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(kc_upper_val) or 
            np.isnan(kc_lower_val) or np.isnan(ema20_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above KC upper with volume
            if close_val > kc_upper_val and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: Break below KC lower with volume
            elif close_val < kc_lower_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below EMA20
            if close_val < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above EMA20
            if close_val > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals