#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Keltner_Channel_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Keltner Channel Width (volatility squeeze) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR (10-period)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # EMA (20-period)
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel Width = (Upper - Lower) / EMA
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    kc_width = (kc_upper - kc_lower) / np.where(ema_20 > 0, ema_20, np.nan)
    
    # Bollinger Band Width (20, 2) for squeeze detection
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, np.nan)
    
    # Squeeze condition: KC width < BB width (volatility contraction)
    squeeze = kc_width < bb_width
    
    # Align 1d indicators to 4h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    
    # === 4h: Volume confirmation ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        squeeze_val = squeeze_aligned[i]
        kc_upper_val = kc_upper_aligned[i]
        kc_lower_val = kc_lower_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(squeeze_val) or np.isnan(kc_upper_val) or 
            np.isnan(kc_lower_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper KC after squeeze + volume
            if (squeeze_val and                    # Volatility squeeze
                close_val > kc_upper_val and       # Break above upper KC
                vol_ratio_val > 1.8):              # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Break below lower KC after squeeze + volume
            elif (squeeze_val and                   # Volatility squeeze
                  close_val < kc_lower_val and      # Break below lower KC
                  vol_ratio_val > 1.8):             # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below middle KC (EMA20) or volatility expansion
            bb_mid_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().iloc[i]
            if close_val < bb_mid_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above middle KC (EMA20) or volatility expansion
            bb_mid_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().iloc[i]
            if close_val > bb_mid_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals