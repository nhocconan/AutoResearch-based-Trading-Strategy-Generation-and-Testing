#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H3/L3) from 4h for breakout signals
- 1d EMA34 as trend filter (long only above, short only below)
- Volume > 1.8x 24-period average for confirmation
- Position size: 0.20 discrete level to minimize fee churn
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Target: 60-150 total trades over 4 years = 15-37/year on 1h timeframe
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels (H3, L3) from prior 4h bar
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    hl_range = high_4h - low_4h
    camarilla_h3 = close_4h + 1.1 * hl_range / 4.0
    camarilla_l3 = close_4h - 1.1 * hl_range / 4.0
    
    # Align Camarilla levels to 1h timeframe (using completed 4h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34)  # Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]  # Close above H3
        breakout_down = close[i] < camarilla_l3_aligned[i]  # Close below L3
        
        if position == 0:
            # Long: 4h Camarilla H3 breakout up AND price above 1d EMA34 AND volume confirmation
            if breakout_up and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: 4h Camarilla L3 breakout down AND price below 1d EMA34 AND volume confirmation
            elif breakout_down and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h Camarilla L3 breakdown OR price crosses below 1d EMA34
            if breakout_down or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h Camarilla H3 breakout OR price crosses above 1d EMA34
            if breakout_up or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0