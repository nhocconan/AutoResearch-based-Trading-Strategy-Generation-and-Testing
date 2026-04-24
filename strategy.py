#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend direction.
- EMA(34) > EMA(89) on 1d indicates bullish trend; EMA(34) < EMA(89) indicates bearish trend.
- Entry: Long when price breaks above Donchian(20) upper AND EMA(34) > EMA(89) (bullish breakout in uptrend).
         Short when price breaks below Donchian(20) lower AND EMA(34) < EMA(89) (bearish breakout in downtrend).
         No entries in sideways markets (EMA(34) and EMA(89) close together).
- Exit: Opposite Donchian breakout or EMA cross flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in both bull and bear by only taking breakouts in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:
        return np.zeros(n)
    
    # Calculate EMAs on 1d
    close_1d = pd.Series(df_1d['close'])
    ema_34 = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = close_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89)
    
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
    start_idx = max(90, lookback, 20)  # Need enough 1d bars for EMA(89) and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        ema_89_val = ema_89_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Only take breakouts in the direction of the 1d trend
                if ema_34_val > ema_89_val:  # Bullish trend on 1d
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema_34_val < ema_89_val:  # Bearish trend on 1d
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                # If EMAs are close (sideways market), no entries
        elif position == 1:
            # Long exit: price closes below Donchian mid OR EMA cross flips to bearish
            if curr_close < donchian_mid[i] or ema_34_val < ema_89_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR EMA cross flips to bullish
            if curr_close > donchian_mid[i] or ema_34_val > ema_89_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_89Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0