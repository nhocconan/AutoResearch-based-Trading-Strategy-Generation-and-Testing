#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX/DMI with 4h trend filter and volume confirmation.
# Use ADX > 25 + DI+ > DI- for uptrend, ADX > 25 + DI- > DI+ for downtrend.
# Confirm with 4h EMA(34) trend direction.
# Entry only during 08-20 UTC session.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Works in bull/bear: ADX filters whipsaws, trend filter aligns with higher timeframe.

name = "1h_ADX_DMI_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate ADX/DMI (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smoothed values
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initial values (first 14 periods)
    if n >= 14:
        atr[13] = np.mean(tr[1:14])
        plus_di[13] = 100 * np.mean(plus_dm[1:14]) / atr[13] if atr[13] != 0 else 0
        minus_di[13] = 100 * np.mean(minus_dm[1:14]) / atr[13] if atr[13] != 0 else 0
    
        # Wilder smoothing
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_di[i] = 100 * (plus_di[i-1] * 13 + plus_dm[i]) / (14 * atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_di[i-1] * 13 + minus_dm[i]) / (14 * atr[i]) if atr[i] != 0 else 0
    
    # ADX calculation
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(14, n):
        di_diff = abs(plus_di[i] - minus_di[i])
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
    
    if n >= 27:
        adx[26] = np.mean(dx[14:27])
        for i in range(27, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # 4h EMA(34) for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_4h = ema_34_4h[1:] > ema_34_4h[:-1]
    trend_up_4h = np.concatenate([[False], trend_up_4h])
    
    # Align 4h trend to 1h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h.astype(float))
    
    # Volume confirmation: 20-period volume spike (1.5x EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(trend_up_4h_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: ADX > 25, DI+ > DI-, 4h uptrend, volume confirmation
            if (adx[i] > 25 and 
                plus_di[i] > minus_di[i] and 
                trend_up_4h_aligned[i] > 0.5 and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: ADX > 25, DI- > DI+, 4h downtrend, volume confirmation
            elif (adx[i] > 25 and 
                  minus_di[i] > plus_di[i] and 
                  trend_up_4h_aligned[i] <= 0.5 and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 or DI- > DI+ or 4h downtrend
            if (adx[i] < 20 or 
                minus_di[i] > plus_di[i] or 
                trend_up_4h_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: ADX < 20 or DI+ > DI- or 4h uptrend
            if (adx[i] < 20 or 
                plus_di[i] > minus_di[i] or 
                trend_up_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals