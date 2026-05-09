#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ADX_Trend_Filter_with_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    Trend-following strategy using 1d ADX as primary filter with 4h volume spikes.
    - Long when: ADX(1d) > 25 AND volume spike (>1.5x 20-period avg) AND price > EMA(50)
    - Short when: ADX(1d) > 25 AND volume spike (>1.5x 20-period avg) AND price < EMA(50)
    - Exit when: ADX drops below 20 OR no volume spike
    - Uses discrete position sizing (0.25) to minimize fee churn
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h data
    ema_period = 50
    ema = np.full(n, np.nan)
    if n >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder smoothing
    period = 14
    alpha = 1.0 / period
    atr = np.full(len(tr), np.nan)
    plus_dm_smooth = np.full(len(tr), np.nan)
    minus_dm_smooth = np.full(len(tr), np.nan)
    
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
        
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
    
    # Align ADX to 4h
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (20-period for 4h)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, ema_period, 2*period)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema[i]) or np.isnan(adx_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        # ADX trend filter: strong trend (ADX > 25) or weak trend (ADX < 20)
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        if position == 0:
            # Long: Strong trend + volume spike + price above EMA
            if strong_trend and vol_spike and close[i] > ema[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend + volume spike + price below EMA
            elif strong_trend and vol_spike and close[i] < ema[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weak trend OR no volume spike OR price crosses below EMA
            if weak_trend or not vol_spike or close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weak trend OR no volume spike OR price crosses above EMA
            if weak_trend or not vol_spike or close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals