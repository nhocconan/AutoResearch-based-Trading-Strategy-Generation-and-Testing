#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot H3/L3 breakout with 1d volume spike filter and ATR-based position sizing.
- Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average volume (bullish breakout)
- Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average volume (bearish breakout)
- Position size fixed at 0.25 to balance risk/reward and minimize fee churn
- Exit on opposite Camarilla level touch (H3 for longs, L3 for shorts) or when volume spike condition fails
- Uses 4h primary with 1d HTF for Camarilla calculation and volume confirmation
- Camarilla levels provide mathematically derived support/resistance; volume spike confirms institutional participation
- Fixed sizing reduces churn while maintaining adequate exposure across volatility regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from previous 1d bar
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Calculate 1d volume spike filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.5 * vol_ma_20)
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Fixed position size to minimize fee churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # need 20 bars for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 level AND volume spike
            if close[i] > camarilla_h3_aligned[i] and volume_spike_aligned[i] > 0.5:
                signals[i] = position_size
                position = 1
            # Short: break below L3 level AND volume spike
            elif close[i] < camarilla_l3_aligned[i] and volume_spike_aligned[i] > 0.5:
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: touch L3 level OR volume spike fails
            if close[i] < camarilla_l3_aligned[i] or volume_spike_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: touch H3 level OR volume spike fails
            if close[i] > camarilla_h3_aligned[i] or volume_spike_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_H3L3_1dVolumeSpike_Breakout_v1"
timeframe = "4h"
leverage = 1.0