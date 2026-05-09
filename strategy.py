#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Camarilla R1/S1 breakout with 1d ADX trend filter and volume confirmation.
    - Long: Price breaks above R1 with ADX>25 and volume spike
    - Short: Price breaks below S1 with ADX>25 and volume spike
    - Exit: Price crosses back through Camarilla pivot point
    - Volume spike: current volume > 2.0 x 24-period average (24 * 12h = 12d)
    - Target: 12-37 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # R1 = C + (H-L)*1.12, S1 = C - (H-L)*1.12, PP = (H+L+C)/3
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_pp = np.full(n, np.nan)
    
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            rng = high[i-1] - low[i-1]
            camarilla_r1[i] = close[i-1] + rng * 1.12
            camarilla_s1[i] = close[i-1] - rng * 1.12
            camarilla_pp[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.full(len(tr), np.nan)
    plus_dm_smooth = np.full(len(tr), np.nan)
    minus_dm_smooth = np.full(len(tr), np.nan)
    
    # Initialize first value with simple average
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
        
        # Wilder smoothing
        for i in range(period, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # Calculate DI and DX
    plus_di = np.full(len(tr), np.nan)
    minus_di = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    for i in range(period, len(tr)):
        if atr[i] > 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / atr[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / atr[i])
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX is smoothed DX
    adx = np.full(len(tr), np.nan)
    if len(dx) >= 2*period-1:
        adx[2*period-2] = np.nanmean(dx[period:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align ADX to 12h
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (24-period for 12h = 12 days)
    vol_avg = np.full(n, np.nan)
    for i in range(24, n):
        vol_avg[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2*period, 24)  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(adx_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 24-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        # ADX trend filter: only trade when trending (ADX > 25)
        trending = adx_12h[i] > 25
        
        if position == 0:
            # Long: Break above R1 with trend and volume spike
            if (close[i] > camarilla_r1[i] and trending and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with trend and volume spike
            elif (close[i] < camarilla_s1[i] and trending and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below pivot point
            if close[i] < camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above pivot point
            if close[i] > camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals