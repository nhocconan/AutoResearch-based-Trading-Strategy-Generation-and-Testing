#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + ADX regime filter with 12h EMA200 trend filter
# Uses Williams %R(14) for oversold/overbought signals (long <20, short >80)
# Only takes signals when ADX(14) > 25 (trending regime) to avoid chop
# Trend filter: 12h EMA200 - only long when price > EMA200, short when price < EMA200
# Position size 0.25 to manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fees

name = "6h_12h_williamsr_adx_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data ONCE before loop for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema_200 = np.full(len(df_12h), np.nan)
    multiplier = 2 / (200 + 1)
    if len(df_12h) > 0:
        ema_200[0] = close_12h[0]
        for i in range(1, len(df_12h)):
            ema_200[i] = (close_12h[i] * multiplier) + (ema_200[i-1] * (1 - multiplier))
    
    # Align 12h EMA200 to 6h timeframe
    ema_200_6h = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Calculate Williams %R(14) on 6h data
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate ADX(14) on 6h data
    # First calculate True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    
    if n >= 14:
        # Initial values
        atr[13] = np.mean(tr[1:14])
        plus_dm_smooth[13] = np.mean(plus_dm[1:14])
        minus_dm_smooth[13] = np.mean(minus_dm[1:14])
        
        # Wilder's smoothing
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(14, n):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.full(n, np.nan)
    if n >= 28:  # Need 14 periods for DX + 14 for smoothing
        adx[27] = np.mean(dx[14:28])
        for i in range(28, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(ema_200_6h[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trending = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R > 50 (exit overbought) OR trend filter fails
            if williams_r[i] > 50 or close[i] <= ema_200_6h[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -50 (exit oversold) OR trend filter fails
            if williams_r[i] < -50 or close[i] >= ema_200_6h[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extremes with trend and ADX filters
            if trending:
                # Long entry: Williams %R < -20 (oversold) AND price above 12h EMA200
                if williams_r[i] < -20 and close[i] > ema_200_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > 80 (overbought) AND price below 12h EMA200
                elif williams_r[i] > 80 and close[i] < ema_200_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals