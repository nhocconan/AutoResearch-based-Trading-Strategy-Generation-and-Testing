#!/usr/bin/env python3
# 12h_1d_adx_volume_v2
# Hypothesis: 12h price breaking above/below 1d pivot point resistance/support levels
# (R1/S1) with volume and ADX trend strength confirmation creates high-probability
# breakout trades. Works in both bull/bear markets by trading breakouts in direction
# of prevailing trend. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_adx_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 days for ADX
        return np.zeros(n)
    
    # Calculate daily pivot points from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - low_1d
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - high_1d
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial values
    atr[period] = np.nansum(tr[1:period+1])
    plus_dm_sum = np.nansum(plus_dm[1:period+1])
    minus_dm_sum = np.nansum(minus_dm[1:period+1])
    
    for i in range(period + 1, len(tr)):
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        plus_di[i] = 100 * (plus_dm_sum / atr[i]) if atr[i] != 0 else 0
        minus_di[i] = 100 * (minus_dm_sum / atr[i]) if atr[i] != 0 else 0
    
    # DX and ADX
    dx = np.zeros_like(tr)
    adx = np.zeros_like(tr)
    for i in range(period*2, len(tr)):  # Start after 2*period for ADX
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx[period*2] = np.nansum(dx[period*2-period+1:period*2+1]) / period
    for i in range(period*2 + 1, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Align pivot levels and ADX to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.3x average of last 4 periods (2 days)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    # ADX trend strength filter: ADX > 20 indicates trending market
    adx_filter = adx_1d_aligned > 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 or ADX weakens
            if close[i] < s1_1d_aligned[i] or adx_1d_aligned[i] < 15:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 or ADX weakens
            if close[i] > r1_1d_aligned[i] or adx_1d_aligned[i] < 15:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Need ADX > 20 for trend strength
            if adx_filter[i]:
                # Long entry: price breaks above R1 with volume
                if (close[i] > r1_1d_aligned[i] and 
                    open_prices[i] <= r1_1d_aligned[i] and  # Ensure breakout happened this bar
                    vol_confirm[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below S1 with volume
                elif (close[i] < s1_1d_aligned[i] and 
                      open_prices[i] >= s1_1d_aligned[i] and  # Ensure breakdown happened this bar
                      vol_confirm[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals