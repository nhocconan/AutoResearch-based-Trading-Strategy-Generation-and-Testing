#!/usr/bin/env python3
name = "6h_ADX_DI_Crossover_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # ADX(14) and DI+ / DI- calculation (standard Wilder's)
    period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (same as EMA with alpha=1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(arr[1:period]) if np.any(~np.isnan(arr[1:period])) else 0
        # subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= period:
        # ADX is Wilder's smoothed DX
        adx[period-1] = np.nanmean(dx[period-1:2*period-1]) if np.any(~np.isnan(dx[period-1:2*period-1])) else 0
        for i in range(2*period-1, len(dx)):
            if np.isnan(adx[i-1]) or np.isnan(dx[i]):
                adx[i] = np.nan
            else:
                adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(2*period-1, 34)  # ensure ADX and EMA are valid
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25, DI+ > DI-, price above 1d EMA34
            if adx[i] > 25 and plus_di[i] > minus_di[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25, DI- > DI+, price below 1d EMA34
            elif adx[i] > 25 and minus_di[i] > plus_di[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 or DI- crosses above DI+ (trend weakening)
            if adx[i] < 20 or minus_di[i] > plus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 or DI+ crosses above DI-
            if adx[i] < 20 or plus_di[i] > minus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals