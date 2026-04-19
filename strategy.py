#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
# Donchian captures breakouts from volatility contractions
# 1d volume surge confirms institutional participation
# ADX(14) > 25 filters for trending markets to avoid whipsaws in ranges
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
name = "4h_Donchian20_1dVol_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX(14) for trend strength
    period_adx = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0
    
    # Smoothed ATR, +DM, -DM
    atr = np.zeros(n)
    atr[period_adx-1] = tr[:period_adx].mean()
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    plus_dm_smooth[period_adx-1] = plus_dm[:period_adx].sum()
    minus_dm_smooth[period_adx-1] = minus_dm[:period_adx].sum()
    
    for i in range(period_adx, n):
        atr[i] = (atr[i-1] * (period_adx-1) + tr[i]) / period_adx
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period_adx-1) + plus_dm[i]) / period_adx
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period_adx-1) + minus_dm[i]) / period_adx
    
    # DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period_adx, n):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
    
    # DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    for i in range(period_adx, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # Smoothed ADX
    adx[2*period_adx-1] = dx[period_adx:2*period_adx].mean()
    for i in range(2*period_adx, n):
        adx[i] = (adx[i-1] * (period_adx-1) + dx[i]) / period_adx
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2*period_adx)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume surge + ADX > 25
            if (close[i] > donchian_high[i] and 
                volume[i] > vol_avg_1d_aligned[i] * 1.5 and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume surge + ADX > 25
            elif (close[i] < donchian_low[i] and 
                  volume[i] > vol_avg_1d_aligned[i] * 1.5 and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or ADX weakens
            if (close[i] < donchian_low[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or ADX weakens
            if (close[i] > donchian_high[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals