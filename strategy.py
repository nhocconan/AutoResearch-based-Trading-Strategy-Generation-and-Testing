#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1-day ADX trend filter and volume confirmation.
Breakouts occur when price breaks Donchian(20) channels, filtered by daily ADX > 25
to ensure trending conditions. Volume > 1.5x average confirms breakout strength.
Uses discrete position sizes (±0.25) to minimize churn. Target: 20-40 trades/year.
ATR-based trailing stop (3x ATR) limits drawdown. Works in bull/bear by capturing
volatility expansion in trending markets.
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(len(close_1d))
    minus_dm = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed TR, +DM, -DM (Wilder smoothing)
    period = 14
    tr_period = np.zeros(len(close_1d))
    plus_dm_period = np.zeros(len(close_1d))
    minus_dm_period = np.zeros(len(close_1d))
    
    if len(close_1d) >= period:
        tr_period[period-1] = np.sum(tr_1d[1:period+1])
        plus_dm_period[period-1] = np.sum(plus_dm[1:period+1])
        minus_dm_period[period-1] = np.sum(minus_dm[1:period+1])  # Fixed bug
        
        for i in range(period, len(close_1d)):
            tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr_1d[i]
            plus_dm_period[i] = plus_dm_period[i-1] - (plus_dm_period[i-1] / period) + plus_dm[i]
            minus_dm_period[i] = minus_dm_period[i-1] - (minus_dm_period[i-1] / period) + minus_dm[i]
    
    # Directional Indicators
    plus_di = np.zeros(len(close_1d))
    minus_di = np.zeros(len(close_1d))
    dx = np.zeros(len(close_1d))
    
    for i in range(period-1, len(close_1d)):
        if tr_period[i] > 0:
            plus_di[i] = 100 * (plus_dm_period[i] / tr_period[i])
            minus_di[i] = 100 * (minus_dm_period[i] / tr_period[i])
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX = smoothed DX
    adx_1d = np.zeros(len(close_1d))
    if len(close_1d) >= 2 * period - 1:
        adx_1d[2*period-2] = np.sum(dx[period-1:2*period-1]) / period
        for i in range(2*period-1, len(close_1d)):
            adx_1d[i] = (adx_1d[i-1] * (period - 1) + dx[i]) / period
    
    # Align 1d ADX to 4h timeframe (waits for 1d bar close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period)
    donch_len = 20
    upper_donch = np.full(n, np.nan)
    lower_donch = np.full(n, np.nan)
    
    for i in range(donch_len-1, n):
        upper_donch[i] = np.max(high[i-donch_len+1:i+1])
        lower_donch[i] = np.min(low[i-donch_len+1:i+1])
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    for i in range(period, n):
        if i == period:
            atr[i] = np.mean(tr[1:period+1])
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Donchian (20), ATR (14), ADX (28), volume MA (20)
    start_idx = max(donch_len-1, period, 2*period-1, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_donch[i]) or
            np.isnan(lower_donch[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above upper Donchian in trending market with volume
            if trending and price > upper_donch[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower Donchian in trending market with volume
            elif trending and price < lower_donch[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below lower Donchian or trend weakens or ATR stop
            if price < lower_donch[i] or adx_aligned[i] < 20 or price < (high[i] - 3.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above upper Donchian or trend weakens or ATR stop
            if price > upper_donch[i] or adx_aligned[i] < 20 or price > (low[i] + 3.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0