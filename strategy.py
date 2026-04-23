#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H4/L4 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H4, L4) from 4h for breakout signals (wider bands = fewer false breaks)
- 4h EMA50 as trend filter (long only above, short only below) - avoids whipsaw in choppy markets
- Volume > 1.6x 20-period average for confirmation (filters low-momentum breakouts)
- Position size: 0.20 discrete level to minimize fee churn
- Session filter: 08-20 UTC (avoid low-volume Asian session)
- Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
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
    
    # Volume confirmation: > 1.6x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h data for Camarilla pivot calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels (H4, L4) from prior 4h bar
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H4 = close + (range * 1.1 / 2)
    # L4 = close - (range * 1.1 / 2)
    pivot = (high_4h + low_4h + close_4h) / 3.0
    rng = high_4h - low_4h
    camarilla_h4 = close_4h + (rng * 1.1 / 2.0)
    camarilla_l4 = close_4h - (rng * 1.1 / 2.0)
    
    # Align Camarilla levels to 4h timeframe (using completed 4h bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # 4h data for EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 08-20 UTC (precompute hours index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation (> 1.6x average)
        volume_confirm = volume[i] > 1.6 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h4_aligned[i]  # Close above H4
        breakout_down = close[i] < camarilla_l4_aligned[i]  # Close below L4
        
        if position == 0 and in_session:
            # Long: 4h Camarilla H4 breakout up AND price above 4h EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: 4h Camarilla L4 breakout down AND price below 4h EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h Camarilla L4 breakdown OR price crosses below 4h EMA50
            if breakout_down or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h Camarilla H4 breakout OR price crosses above 4h EMA50
            if breakout_up or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_Breakout_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0