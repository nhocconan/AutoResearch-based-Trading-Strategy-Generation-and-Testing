#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ATR regime filter and volume confirmation.
# Long when price breaks above upper BB(20,2) AND 1d ATR(14) > 1d ATR(50) (high volatility regime) AND volume > 1.5x 20-period average.
# Short when price breaks below lower BB(20,2) AND 1d ATR(14) > 1d ATR(50) AND volume > 1.5x 20-period average.
# Exit when price returns to middle BB(20) OR volatility collapses (1d ATR(14) < 1d ATR(50)).
# Bollinger Squeeze identifies low volatility periods; breakout in high volatility regime captures strong moves.
# Works in both bull and bear markets by trading expansion phases after contraction.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.

name = "6h_BollingerSqueeze_Breakout_1dATR_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # 6h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR(14) and ATR(50) for volatility regime filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # first TR is undefined
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # High volatility regime: ATR(14) > ATR(50)
    high_vol_regime = atr_14 > atr_50
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for 50-period ATR
        # Skip if missing data
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or
            np.isnan(high_vol_regime_aligned[i]) or
            np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper BB AND high vol regime AND volume confirm
            if (close[i] > upper_bb[i] and 
                high_vol_regime_aligned[i] > 0.5 and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower BB AND high vol regime AND volume confirm
            elif (close[i] < lower_bb[i] and 
                  high_vol_regime_aligned[i] > 0.5 and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle BB OR volatility collapses
            if (close[i] < middle_bb[i] or 
                high_vol_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle BB OR volatility collapses
            if (close[i] > middle_bb[i] or 
                high_vol_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals