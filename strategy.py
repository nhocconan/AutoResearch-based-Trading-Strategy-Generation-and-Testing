#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) combined with 1d Alligator (SMAs) filter and volume confirmation.
- Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
- Alligator uses SMAs (13,8,5) smoothed; aligned teeth (SMA8) acts as dynamic trend filter.
- Long when Bull Power > 0, price > SMA8 (teeth), and volume > 2.0x 24-period average.
- Short when Bear Power < 0, price < SMA8 (teeth), and volume > 2.0x 24-period average.
- Uses discrete position size 0.25 to manage drawdown and reduce fee churn.
- Volume confirmation ensures conviction; avoids low-momentum false signals.
- Designed for 12-30 trades/year (50-120 total over 4 years) to stay within fee-efficient range.
- Combines trend (Alligator) and momentum (Elder Ray) with volume filter for robustness in bull/bear markets.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed daily bar)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 6h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate EMA13 on 1d close for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d_aligned - ema_13_1d_aligned
    bear_power = low_1d_aligned - ema_13_1d_aligned
    
    # Alligator: SMA(13), SMA(8), SMA(5) on 1d close, all smoothed
    smi_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    smi_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    smi_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Smoothed (alligator jaws/teeth/lips) - additional 3-period SMAs
    smi_13_smooth = pd.Series(smi_13).rolling(window=3, min_periods=3).mean().values
    smi_8_smooth = pd.Series(smi_8).rolling(window=3, min_periods=3).mean().values
    smi_5_smooth = pd.Series(smi_5).rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator components (teeth = SMA8 smoothed is key trend filter)
    smi_13_aligned = align_htf_to_ltf(prices, df_1d, smi_13_smooth)
    smi_8_aligned = align_htf_to_ltf(prices, df_1d, smi_8_smooth)  # Teeth
    smi_5_aligned = align_htf_to_ltf(prices, df_1d, smi_5_smooth)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(smi_8_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price > Alligator teeth (SMA8) AND volume confirmation
            if bull_power[i] > 0 and close[i] > smi_8_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price < Alligator teeth (SMA8) AND volume confirmation
            elif bear_power[i] < 0 and close[i] < smi_8_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR price crosses below Alligator teeth
            if bull_power[i] <= 0 or close[i] < smi_8_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR price crosses above Alligator teeth
            if bear_power[i] >= 0 or close[i] > smi_8_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Alligator_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0