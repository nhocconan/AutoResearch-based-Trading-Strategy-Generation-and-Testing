#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike (>2x 24-bar average) + session filter (08-20 UTC).
- Uses 4h for entry timing and 1d for trend filter (proven pattern from top performers)
- Volume spike reduces false breakouts in low volatility
- Session filter avoids low-liquidity periods
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 20-50 trades/year (80-200 over 4 years) to stay within fee drag limits
- Works in bull/bear via trend filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 2.0x 24-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 4h Donchian(20) breakout levels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24, 20)  # EMA34, volume MA, Donchian
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_roll[i]) or
            np.isnan(low_roll[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > high_roll[i-1]  # Close above prior 20-bar high
        breakout_down = close[i] < low_roll[i-1]  # Close below prior 20-bar low
        
        if position == 0:
            # Long: Donchian breakout up AND price > 1d EMA34 AND volume confirmation AND in session
            if breakout_up and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND price < 1d EMA34 AND volume confirmation AND in session
            elif breakout_down and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian break down OR price < 1d EMA34 (trend flip)
            if close[i] < low_roll[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian break up OR price > 1d EMA34 (trend flip)
            if close[i] > high_roll[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_Session"
timeframe = "4h"
leverage = 1.0