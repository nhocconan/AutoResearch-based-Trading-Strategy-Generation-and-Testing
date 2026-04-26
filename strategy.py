#!/usr/bin/env python3
"""
6h_ADX_DMI_TrendStrength_v1
Hypothesis: Trade 6h bars when ADX > 25 indicates strong trend, using +DI/-DI crossover for direction.
In bull markets: ADX > 25 and +DI crosses above -DI → long trend continuation.
In bear markets: ADX > 25 and -DI crosses above +DI → short trend continuation.
1d EMA50 filter ensures alignment with higher timeframe trend to reduce counter-trend trades.
Volume confirmation requires current volume > 1.5 * 20-period average to ensure participation.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX and DMI on 6h data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial TR and DM sums
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    
    # First value: simple average of first 'period' values
    tr_sum[period] = np.nansum(tr[1:period+1])
    plus_dm_sum[period] = np.nansum(plus_dm[1:period+1])
    minus_dm_sum[period] = np.nansum(minus_dm[1:period+1])
    
    # Wilder's smoothing: subsequent values
    for i in range(period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_sum / np.where(tr_sum == 0, np.nan, tr_sum)
    minus_di = 100 * minus_dm_sum / np.where(tr_sum == 0, np.nan, tr_sum)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    
    # ADX: Wilder's smoothing of DX
    adx = np.zeros_like(dx)
    adx[2*period] = np.nanmean(dx[period+1:2*period+1])  # First ADX after 2*period
    
    for i in range(2*period + 1, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Align 1d EMA50 to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ADX, DMI, EMA, and volume MA
    start_idx = max(2*period + 1, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or
            np.isnan(plus_di[i]) or
            np.isnan(minus_di[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        adx_strong = adx[i] > 25
        di_cross_up = plus_di[i] > minus_di[i]
        di_cross_down = minus_di[i] > plus_di[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: ADX > 25, +DI crosses above -DI, volume confirm, 1d uptrend
            long_signal = adx_strong and di_cross_up and vol_conf and trend_up
            
            # Short: ADX > 25, -DI crosses above +DI, volume confirm, 1d downtrend
            short_signal = adx_strong and di_cross_down and vol_conf and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ADX weakens (< 20) OR -DI crosses above +DI (trend reversal) OR 1d trend flips down
            if (adx[i] < 20) or (minus_di[i] > plus_di[i]) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ADX weakens (< 20) OR +DI crosses above -DI (trend reversal) OR 1d trend flips up
            if (adx[i] < 20) or (plus_di[i] > minus_di[i]) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_TrendStrength_v1"
timeframe = "6h"
leverage = 1.0