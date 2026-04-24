#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR volatility filter and volume spike confirmation.
- Uses Camarilla pivot levels from prior 1d to identify key support/resistance (H3/L3).
- Breakout above H3 or below L3 with volume > 2x 20-bar average signals institutional participation.
- Volatility filter: current ATR(14) must be > 0.5x 1d ATR(14) to avoid low-volatility chop.
- Designed for 4h timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (80-200 total over 4 years) to stay fee-efficient.
"""

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC for Camarilla (completed 1d bar)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: H3, L3
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d ATR14 volatility filter
    high_low_1d = np.abs(df_1d['high'].values - df_1d['low'].values)
    high_close_1d = np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    low_close_1d = np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h ATR14 for volatility comparison
    high_low = np.abs(high - low)
    high_close = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        # Volatility filter: current 4h ATR > 0.5x 1d ATR
        vol_filter = atr_14[i] > 0.5 * atr_14_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above H3 AND volume confirmation AND volatility filter
            if close[i] > camarilla_h3_aligned[i] and volume_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L3 AND volume confirmation AND volatility filter
            elif close[i] < camarilla_l3_aligned[i] and volume_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below L3
            if close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above H3
            if close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0