#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ADX > 25 AND 6h volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND 6h volume > 1.5x 20-period average.
# Exit when price crosses the 20-period Donchian midpoint (mean reversion signal).
# Uses Donchian breakouts for trend capture with ADX filter to avoid choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drag.

name = "6h_DonchianBreakout_1dADX25_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d ADX (14-period) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # Initial values (simple average of first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[1:14])
        plus_di[13] = np.mean(plus_dm[1:14]) * 100 / atr[13] if atr[13] != 0 else 0
        minus_di[13] = np.mean(minus_dm[1:14]) * 100 / atr[13] if atr[13] != 0 else 0
        
        # Wilder's smoothing for subsequent periods
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_di[i] = (plus_di[i-1] * 13 + plus_dm[i]) * 100 / atr[i] if atr[i] != 0 else plus_di[i-1]
            minus_di[i] = (minus_di[i-1] * 13 + minus_dm[i]) * 100 / atr[i] if atr[i] != 0 else minus_di[i-1]
    
    # Calculate ADX
    dx = np.zeros_like(atr)
    for i in range(len(atr)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
    
    adx = np.zeros_like(dx)
    if len(dx) >= 27:  # Need 14 + 13 periods for ADX
        adx[26] = np.mean(dx[14:27])
        for i in range(27, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high, strong trend (ADX>25), volume spike
            long_cond = (close[i] > highest_high[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: break below Donchian low, strong trend (ADX>25), volume spike
            short_cond = (close[i] < lowest_low[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i] and close[i-1] >= donchian_mid[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i] and close[i-1] <= donchian_mid[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals