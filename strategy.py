#!/usr/bin/env python3
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
    
    # Get 1d data for ADX and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 28:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1d[i] = np.mean(tr[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed +DM and -DM (14-period)
    plus_dm_smooth = np.full(len(plus_dm), np.nan)
    minus_dm_smooth = np.full(len(minus_dm), np.nan)
    for i in range(14, len(plus_dm)):
        if i == 14:
            plus_dm_smooth[i] = np.sum(plus_dm[1:15])
            minus_dm_smooth[i] = np.sum(minus_dm[1:15])
        else:
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = np.full(len(plus_dm_smooth), np.nan)
    minus_di = np.full(len(minus_dm_smooth), np.nan)
    for i in range(14, len(plus_dm_smooth)):
        if not np.isnan(atr_1d[i]) and atr_1d[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_1d[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_1d[i]
    
    # Calculate DX and ADX (14-period)
    dx = np.full(len(plus_di), np.nan)
    for i in range(14, len(plus_di)):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx_1d = np.full(len(dx), np.nan)
    for i in range(28, len(dx)):  # Need 14 periods of DX
        if i == 28:
            adx_1d[i] = np.mean(dx[14:28])
        else:
            adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200
    ema_period = 200
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / (ema_period + 1)) + 
                        ema_4h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need ADX, EMA, and volume MA
    start_idx = max(28, 200, vol_period) + 5  # extra buffer for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        adx = adx_1d_aligned[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending) + volume spike + price > 4h EMA200
            if (adx > 25 and 
                vol_ratio > 1.5 and 
                price > ema_4h_aligned[i]):
                signals[i] = size
                position = 1
            # Short: ADX > 25 (trending) + volume spike + price < 4h EMA200
            elif (adx > 25 and 
                  vol_ratio > 1.5 and 
                  price < ema_4h_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: ADX < 20 (trend weakening) OR price < EMA200
            if (adx < 20 or 
                price < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: ADX < 20 (trend weakening) OR price > EMA200
            if (adx < 20 or 
                price > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_ADX_Trend_VolumeSpike_EMA200"
timeframe = "1h"
leverage = 1.0