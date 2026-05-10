#!/usr/bin/env python3
"""
1h_ADX_Regime_ADX14_DI_Cross
Hypothesis: Use ADX(14) to filter regime and DI crossover for entry in 1h timeframe.
Long when +DI crosses above -DI AND ADX > 25 (trending up).
Short when -DI crosses above +DI AND ADX > 25 (trending down).
Exit when ADX falls below 20 (range) or opposite DI cross.
Uses 4h EMA50 as higher timeframe trend filter: only take longs when price > EMA50_4h, shorts when price < EMA50_4h.
Adds session filter (08-20 UTC) to avoid low-volume Asian session.
Target: 15-35 trades/year (~60-140 over 4 years) to minimize fee drag.
Works in bull (trend following) and bear (trend following) markets.
"""

name = "1h_ADX_Regime_ADX14_DI_Cross"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) to avoid low-volume sessions
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # ADX calculation (14 period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0
        
        # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
            minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        dx = np.zeros_like(atr)
        mask = (plus_dm_smooth + minus_dm_smooth) > 0
        dx[mask] = 100 * np.abs(plus_dm_smooth[mask] - minus_dm_smooth[mask]) / (plus_dm_smooth[mask] + minus_dm_smooth[mask])
        
        # ADX = smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            # First ADX is average of first 'period' DX values
            valid_dx = dx[period-1:2*(period-1)+1]  # DX[13] to DX[27] for period=14
            if len(valid_dx) >= period and not np.all(np.isnan(valid_dx)):
                adx[2*(period-1)] = np.nanmean(valid_dx)  # ADX[27] for period=14
                for i in range(2*(period-1)+1, len(dx)):
                    if not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
                    else:
                        adx[i] = adx[i-1]
        return adx, plus_dm_smooth, minus_dm_smooth
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for ADX calculation
    start_idx = max(40, 1)  # Need ADX(14) to stabilize
    
    for i in range(start_idx, n):
        if (not in_session[i] or 
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: +DI crosses above -DI, ADX > 25 (strong uptrend), price above 4h EMA50
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and  # Cross above
                adx[i] > 25 and close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: -DI crosses above +DI, ADX > 25 (strong downtrend), price below 4h EMA50
            elif (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and  # Cross above
                  adx[i] > 25 and close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 (weakening trend) OR -DI crosses above +DI
            if (adx[i] < 20 or 
                (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (weakening trend) OR +DI crosses above -DI
            if (adx[i] < 20 or 
                (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals