#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1w EMA50 trend filter and ATR-based volume confirmation.
- Uses Camarilla pivot levels (H4, L4) from 1d for stronger breakout signals (less frequent than H3/L3)
- 1w EMA50 as trend filter for multi-timeframe alignment (long only above, short only below)
- Volume > 2.0x ATR-scaled average for confirmation (adjusts to volatility)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
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
    
    # ATR for volatility-adjusted volume confirmation
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    vol_norm = volume / (close * atr_safe / 100)  # Volume normalized by ATR% of price
    vol_ma = pd.Series(vol_norm).rolling(window=24, min_periods=24).mean().values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (H4, L4) from prior 1d bar
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    camarilla_h4 = close_1d + (rng * 1.1 / 2.0)  # H4 level
    camarilla_l4 = close_1d - (rng * 1.1 / 2.0)  # L4 level
    
    # Align Camarilla levels to 1d timeframe (using completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14, 50)  # Volume MA, ATR, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x ATR-normalized average)
        volume_confirm = vol_norm[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h4_aligned[i]  # Close above H4
        breakout_down = close[i] < camarilla_l4_aligned[i]  # Close below L4
        
        if position == 0:
            # Long: 1d Camarilla H4 breakout up AND price above 1w EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1d Camarilla L4 breakout down AND price below 1w EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d Camarilla L4 breakdown OR price crosses below 1w EMA50
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d Camarilla H4 breakout OR price crosses above 1w EMA50
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1wEMA50_ATRVolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0