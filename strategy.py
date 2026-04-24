#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR-based volatility filter and volume spike confirmation.
- Uses Camarilla pivot levels (H3, L3) from 1d timeframe as strong intraday support/resistance.
- Breakout above H3 with volume > 1.8x 20-bar average = long signal.
- Breakdown below L3 with volume > 1.8x 20-bar average = short signal.
- Volatility filter: 1d ATR(14) must be > 0.5x its 50-period average to ensure sufficient momentum.
- Designed for 4h timeframe to capture swings with higher probability entries.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (80-200 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false breakouts in choppy markets.
- Novelty: Uses H3/L3 levels (more frequently tested than H4/L4) with ATR volatility filter on 4h timeframe.
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
    
    # Get 1d data ONCE before loop for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4  # H3 level
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4  # L3 level
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d ATR(14) volatility filter
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # Align with close_1d index
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14_1d > 0.5 * atr_ma_50_1d  # ATR > 50% of its 50-period MA
    
    # Align volatility filter to 4h timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter.astype(float))
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough for ATR MA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(atr_filter_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average) and volatility filter
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        vol_filter = atr_filter_aligned[i] > 0.5  # Boolean as float
        
        if position == 0:
            # Only trade if volume confirms breakout and volatility is sufficient
            if volume_confirm and vol_filter:
                # Long: price breaks above H3
                if close[i] > h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below L3
                elif close[i] < l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below L3
            if close[i] < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above H3
            if close[i] > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0