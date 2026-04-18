#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Hypothesis: Camarilla pivot levels (R1/S1) from 1d provide institutional support/resistance. 
Breakouts with volume confirmation and ADX > 25 trend filter capture real moves. 
Position size 0.25 balances risk/reward. Works in bull (breakouts up) and bear (breakouts down).
Target: 20-40 trades/year by requiring multiple confirmations.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) for each 1d bar
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # ADX(14) for trend strength on 12h data
    # +DM = max(0, high - high_prev)
    # -DM = max(0, low_prev - low)
    # TR = max(high - low, high - close_prev, low - close_prev)
    # +DI = 100 * EMA(+DM) / ATR
    # -DI = 100 * EMA(-DM) / ATR
    # ADX = EMA(|+DI - -DI| / (+DI + -DI))
    period = 14
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    plus_dm = np.where((high - high_prev) > (low_prev - low), np.maximum(high - high_prev, 0), 0)
    minus_dm = np.where((low_prev - low) > (high - high_prev), np.maximum(low_prev - low, 0), 0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    adx = np.full_like(tr, np.nan)
    
    if len(tr) >= period:
        # Initial values
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sum = np.sum(plus_dm[:period])
        minus_dm_sum = np.sum(minus_dm[:period])
        plus_di[period-1] = 100 * plus_dm_sum / (atr[period-1] * period) if atr[period-1] != 0 else 0
        minus_di[period-1] = 100 * minus_dm_sum / (atr[period-1] * period) if atr[period-1] != 0 else 0
        
        # Wilder smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_di[i] = 100 * ((plus_di[i-1] * (period - 1) + plus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * ((minus_di[i-1] * (period - 1) + minus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
            
        # ADX smoothing
        if len(dx) >= 2 * period - 1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2*period-1, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx[i] > 25
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and trending:
            # Long: price breaks above R1 with volume
            if close[i] > r1_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif close[i] < s1_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0