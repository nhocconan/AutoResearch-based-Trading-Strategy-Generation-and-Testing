#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend direction.
- EMA > rising indicates bullish trend, EMA < falling indicates bearish trend.
- Entry: Long when price breaks above Donchian(20) upper AND 12h EMA(34) rising (bullish breakout in uptrend).
         Short when price breaks below Donchian(20) lower AND 12h EMA(34) falling (bearish breakout in downtrend).
- Exit: Opposite Donchian breakout or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h
    ema_34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, lookback, 20)  # Need enough 12h bars for EMA and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine EMA trend: rising (bullish) or falling (bearish)
        if i >= 1:
            ema_prev = ema_34_aligned[i-1]
            ema_curr = ema_34_aligned[i]
            ema_rising = ema_curr > ema_prev
            ema_falling = ema_curr < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price closes above upper Donchian AND EMA rising
                if curr_close > highest_high[i] and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian AND EMA falling
                elif curr_close < lowest_low[i] and ema_falling:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR EMA starts falling
            if curr_close < donchian_mid[i] or ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR EMA starts rising
            if curr_close > donchian_mid[i] or ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0