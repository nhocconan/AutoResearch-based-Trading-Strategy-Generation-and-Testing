#!/usr/bin/env python3
"""
6h_ADX_DMI_VolumeSpike_1dTrend
Hypothesis: Combine ADX(14) for trend strength with DMI crossover for direction on 6h timeframe, filtered by 1d EMA50 trend and volume spikes (>1.8x 20-period MA). 
Long when +DI crosses above -DI with ADX>25 and 1d uptrend and volume spike. 
Short when -DI crosses above +DI with ADX>25 and 1d downtrend and volume spike.
Exit when ADX<20 (trend weakens) or opposite DMI crossover occurs.
Uses discrete position sizing (0.25) to minimize fee churn.
ADX filters out ranging markets, making it effective in both bull and bear trends.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # 6h ADX/DMI calculation (Wilder's smoothing)
    period = 14
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    up_move = np.concatenate([[np.nan], high[1:] - high[:-1]])
    down_move = np.concatenate([[np.nan], low[:-1] - low[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    alpha = 1.0 / period
    tr_smoothed = np.zeros_like(tr)
    plus_dm_smoothed = np.zeros_like(plus_dm)
    minus_dm_smoothed = np.zeros_like(minus_dm)
    # First value is simple average
    tr_smoothed[period] = np.nansum(tr[1:period+1])
    plus_dm_smoothed[period] = np.nansum(plus_dm[1:period+1])
    minus_dm_smoothed[period] = np.nansum(minus_dm[1:period+1])
    # Subsequent values: Wilder's smoothing
    for i in range(period+1, len(tr)):
        tr_smoothed[i] = tr_smoothed[i-1] - (tr_smoothed[i-1] / period) + tr[i]
        plus_dm_smoothed[i] = plus_dm_smoothed[i-1] - (plus_dm_smoothed[i-1] / period) + plus_dm[i]
        minus_dm_smoothed[i] = minus_dm_smoothed[i-1] - (minus_dm_smoothed[i-1] / period) + minus_dm[i]
    # +DI and -DI
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # ADX = EMA of DX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.nanmean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Volume confirmation: volume > 1.8x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 2*period for ADX + 20 for volume MA)
    start_idx = 2 * period + 20  # 48
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: +DI crosses above -DI with ADX>25 and 1d uptrend and volume spike
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and 
                adx[i] > 25 and uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: -DI crosses above +DI with ADX>25 and 1d downtrend and volume spike
            elif (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and 
                  adx[i] > 25 and downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ADX<20 (trend weakens) OR -DI crosses above +DI
            if (adx[i] < 20 or (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1])):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ADX<20 (trend weakens) OR +DI crosses above -DI
            if (adx[i] < 20 or (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1])):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0