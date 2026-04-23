#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR-based volatility filter and volume spike confirmation.
- Long: Close > Camarilla H3 AND 1d ATR(14) > 1.5x 50-period ATR MA AND volume > 2.0x 20-period avg
- Short: Close < Camarilla L3 AND 1d ATR(14) > 1.5x 50-period ATR MA AND volume > 2.0x 20-period avg
- Exit: Opposite Camarilla breakout OR 1d ATR(14) < 1.0x 50-period ATR MA (volatility collapse)
- Uses 1d HTF for ATR volatility filter and Camarilla levels (calculated from prior completed bars)
- Designed for low trade frequency (19-50/year) to minimize fee drag on 4h timeframe
- Camarilla H3/L3 provide stronger structure than H4/L4, reducing false breakouts
- Volatility filter ensures trades only during sufficient market movement (works in bull/bear)
- Volume confirmation filters low-conviction moves
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for volatility filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50-period MA of 1d ATR for volatility regime filter
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla levels from prior 1d bar (HTF = 1d)
    # Standard Camarilla: H3 = close + 1.25*(high-low), L3 = close - 1.25*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.25 * range_1d
    camarilla_l3 = close_1d - 1.25 * range_1d
    
    # Align indicators to 4h timeframe (use prior completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR MA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_ma_50_1d_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: current ATR > 1.5x ATR MA (sufficient volatility)
        volatility_filter = atr_14_1d_aligned[i] > 1.5 * atr_ma_50_1d_aligned[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Close above prior H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Close below prior L3
        
        if position == 0:
            # Long: Camarilla H3 breakout up AND volatility filter AND volume confirmation
            if breakout_up and volatility_filter and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla L3 breakout down AND volatility filter AND volume confirmation
            elif breakout_down and volatility_filter and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla L3 breakout down OR volatility collapse (ATR < 1.0x ATR MA)
            if breakout_down or atr_14_1d_aligned[i] < 1.0 * atr_ma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla H3 breakout up OR volatility collapse (ATR < 1.0x ATR MA)
            if breakout_up or atr_14_1d_aligned[i] < 1.0 * atr_ma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0