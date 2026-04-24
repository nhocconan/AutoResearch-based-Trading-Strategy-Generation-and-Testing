#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 Breakout with 1d Volume Spike and ATR Trend Filter.
- H4/L4 levels are stronger breakout points than H3/L3, reducing false breakouts.
- 1d ATR(14) > 20-period average indicates high volatility regime, favoring breakout strategies.
- Volume spike (>1.8x 24-period average) confirms breakout validity.
- Discrete position sizing (0.25) balances return potential with fee minimization.
- Target: 50-120 trades over 4 years to avoid fee drag while capturing strong moves.
- Works in bull/bear markets via volatility regime filter and volume confirmation.
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
    
    # Get 1d data ONCE before loop for Camarilla levels, ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla H4 and L4 levels (stronger breakout points)
        camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
        camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
        
        # Align Camarilla levels to 12h timeframe (using previous completed 1d bar)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    else:
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
    
    # 1d ATR(14) trend filter - high volatility regime
    if len(df_1d) >= 15:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range calculation
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
        high_volatility = atr_14 > atr_ma_20  # ATR above its 20-period average
        
        # Align volatility filter to 12h timeframe
        high_volatility_aligned = align_htf_to_ltf(prices, df_1d, high_volatility.astype(float))
    else:
        high_volatility_aligned = np.zeros(n)
    
    # Volume confirmation: > 1.8x 24-period average volume (12h * 2 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H4 with volume spike and high volatility regime
            if close[i] > camarilla_h4_aligned[i] and volume_spike[i] and high_volatility_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: break below L4 with volume spike and high volatility regime
            elif close[i] < camarilla_l4_aligned[i] and volume_spike[i] and high_volatility_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below L4 OR volatility drops (regime change)
            if close[i] < camarilla_l4_aligned[i] or high_volatility_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H4 OR volatility drops (regime change)
            if close[i] > camarilla_h4_aligned[i] or high_volatility_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_Breakout_1dATR_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0