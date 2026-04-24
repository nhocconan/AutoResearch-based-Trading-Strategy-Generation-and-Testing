#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d EMA trend filter and volume spike.
- Williams %R(14) identifies overbought/oversold conditions (< -80 = oversold, > -20 = overbought)
- In trending markets (price > 1d EMA50): fade extremes for mean reversion entries
- In ranging markets (price near 1d EMA50): breakout Donchian(20) for trend continuation
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter noise
- Discrete signal size: 0.25 for risk control
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
- Works in both bull/bear: mean reversion in trends, breakouts in ranges adapts to regime
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
    
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R (14-period) on 4h
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Donchian channels (20-period) for breakout signals
    lookback_dc = 20
    highest_high_dc = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low_dc = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    donchian_mid = (highest_high_dc + lowest_low_dc) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback_wr, lookback_dc, 20)  # Need EMA50 and lookbacks
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high_dc[i]) or np.isnan(lowest_low_dc[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        wr_val = williams_r[i]
        
        if position == 0:
            # Check for entry signals based on regime
            if volume_spike[i]:
                # Trending regime: price away from 1d EMA50 -> mean reversion at extremes
                if abs(curr_close - ema_50_val) / ema_50_val > 0.02:  # >2% deviation = trending
                    # Long when oversold and reversing up
                    if wr_val < -80 and curr_close > prev_close:
                        signals[i] = 0.25
                        position = 1
                    # Short when overbought and reversing down
                    elif wr_val > -20 and curr_close < prev_close:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime: price near 1d EMA50 -> breakout continuation
                    # Long breakout: price closes above upper Donchian
                    if curr_close > highest_high_dc[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short breakout: price closes below lower Donchian
                    elif curr_close < lowest_low_dc[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R overbought OR price crosses Donchian mid
            if wr_val > -20 or curr_close < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R oversold OR price crosses Donchian mid
            if wr_val < -80 or curr_close > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0