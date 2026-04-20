#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_AdaptiveVolatilityBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily ATR-based volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) with proper initialization
    atr_series = pd.Series(tr)
    atr_14 = atr_series.rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: low vol = contraction, high vol = expansion
    atr_ma30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    vol_ratio = atr_14 / np.where(atr_ma30 > 0, atr_ma30, np.nan)
    
    # Align to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio with proper initialization
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio_current = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Bollinger Bands (20, 2) on 6h for breakout detection
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio_aligned[i]
        vol_ratio_current_val = vol_ratio_current[i]
        upper_bb_val = upper_bb[i]
        lower_bb_val = lower_bb[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(vol_ratio_current_val) or 
            np.isnan(upper_bb_val) or np.isnan(lower_bb_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger breakout with low volatility regime and volume confirmation
            if (close_val > upper_bb_val and 
                vol_ratio_val < 0.8 and  # Low volatility contraction
                vol_ratio_current_val > 1.8):  # Volume expansion
                signals[i] = 0.25
                position = 1
            # Short: Bollinger breakdown with low volatility regime and volume confirmation
            elif (close_val < lower_bb_val and 
                  vol_ratio_val < 0.8 and  # Low volatility contraction
                  vol_ratio_current_val > 1.8):  # Volume expansion
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to middle band or volatility expands too much
            if close_val < sma20[i] or vol_ratio_val > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to middle band or volatility expands too much
            if close_val > sma20[i] or vol_ratio_val > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals