#!/usr/bin/env python3
"""
Hypothesis: 1h 4-hour Camarilla H4/L4 breakout with daily EMA50 trend filter and volume spike confirmation.
- Long: Close > Camarilla H4 AND price > daily EMA50 AND volume > 2.5x 20-period avg
- Short: Close < Camarilla L4 AND price < daily EMA50 AND volume > 2.5x 20-period avg
- Exit: Opposite Camarilla breakout OR price crosses daily EMA50
- Uses 4h HTF for Camarilla levels and 1d HTF for EMA50 to reduce noise
- Designed for low trade frequency (15-37/year) to minimize fee drag on 1h timeframe
- Session filter (08-20 UTC) to avoid low-liquidity hours
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
    
    # Volume confirmation: > 2.5x 20-period average (20*1h = 20 hours)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 4h bar (HTF = 4h)
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_h4 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_l4 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Align Camarilla levels to 1h timeframe (use prior completed 4h bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.5x average)
        volume_confirm = volume[i] > 2.5 * vol_ma[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up = close[i] > camarilla_h4_aligned[i-1]  # Close above prior H4
        breakout_down = close[i] < camarilla_l4_aligned[i-1]  # Close below prior L4
        
        if position == 0:
            # Long: Camarilla H4 breakout up AND price > daily EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla L4 breakout down AND price < daily EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Camarilla L4 breakout down OR price < daily EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Camarilla H4 breakout up OR price > daily EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_Breakout_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0