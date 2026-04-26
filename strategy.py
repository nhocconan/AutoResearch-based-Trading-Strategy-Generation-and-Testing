#!/usr/bin/env python3
"""
6h_ADX_DI_Cross_12hTrend_VolumeFilter
Hypothesis: 6h timeframe with ADX(14) DI+ crossing above DI- for long entries, DI- crossing above DI+ for shorts, only in direction of 12h EMA50 trend, confirmed by volume > 1.5x 20-bar MA. Uses discrete sizing (0.25) to limit churn. Designed to work in both bull/bear via trend alignment and ADX filtering for trending markets only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ADX(14) and DI+/- on 6h data
    period = 14
    if len(high) < period + 1:
        return np.zeros(n)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    
    # Wilder smoothing
    alpha = 1.0 / period
    for i in range(period+1, n):
        atr[i] = atr[i-1] * (1 - alpha) + alpha * tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + alpha * plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + alpha * minus_dm[i]
    
    # DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros(n)
    if len(dx) >= 2*period + 1:
        adx[2*period] = np.mean(dx[period+1:2*period+1])
    
    for i in range(2*period+1, n):
        adx[i] = adx[i-1] * (1 - alpha) + alpha * dx[i]
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations
    start_idx = max(2*period+1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(adx[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_12h_val = ema_50_12h_aligned[i]
        vol_ok = volume_filter[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_12h_val
        bearish_12h = close_val < ema_12h_val
        
        # ADX trend strength filter (ADX > 20 for trending market)
        trending = adx[i] > 20
        
        # DI crossover signals
        di_cross_up = plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]
        di_cross_down = minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]
        
        # Entry conditions
        long_entry = di_cross_up and bullish_12h and vol_ok and trending
        short_entry = di_cross_down and bearish_12h and vol_ok and trending
        
        # Exit conditions: opposite DI crossover or loss of trend/volume
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (di_cross_down or not bullish_12h or not vol_ok or not trending):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (di_cross_up or not bearish_12h or not vol_ok or not trending):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ADX_DI_Cross_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0