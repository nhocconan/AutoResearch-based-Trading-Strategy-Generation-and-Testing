#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H3, L3) from prior completed 1d candles for stronger breakout signals.
- Breakout above H3 or below L3 with volume > 2.0x 20-bar average signals strong momentum.
- Trend filter: price must be above/below 1d EMA50 to align with higher timeframe direction.
- Designed for 4h timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (75-200 total over 4 years) to stay fee-efficient.
- H3/L3 levels provide wider bands than R1/S1, reducing false breakouts while capturing strong moves.
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
    
    # Get 1d data ONCE before loop for Camarilla pivots and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from prior completed 1d candles
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla width
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0
    # H3 and L3 levels (stronger breakout levels)
    h3_1d = close_1d + camarilla_width * 1.5
    l3_1d = close_1d - camarilla_width * 1.5
    
    # Align H3 and L3 to 4h timeframe (wait for 1d bar to close)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above H3 AND price above 1d EMA50 AND volume confirmation
            if close[i] > h3_1d_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L3 AND price below 1d EMA50 AND volume confirmation
            elif close[i] < l3_1d_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below L3 OR price below 1d EMA50
            if close[i] < l3_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above H3 OR price above 1d EMA50
            if close[i] > h3_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0