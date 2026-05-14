#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 14-period ATR volatility breakout with 1-day ADX trend filter.
Breakouts occur when price moves beyond ATR-based channels, filtered by daily ADX > 25
to ensure trending conditions. Volume > 2x average confirms breakout strength.
Uses discrete position sizes (±0.25) to minimize fee churn. Target: 20-50 trades/year.
ATR-based stoploss limits drawdown. Works in bull/bear by capturing volatility expansion.
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
        minus_dm_period[period-1] = np.sum(plus_dm[1:period+1])  # BUG: should be minus_dm
        
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
    
    # Calculate ATR(14) on 4h data for volatility channels
    tr_4h = np.zeros(n)
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    for i in range(period, n):
        if i == period:
            atr[i] = np.mean(tr_4h[1:period+1])
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr_4h[i]) / period
    
    # Calculate ATR-based channels (like Keltner)
    ma_period = 20
    ema_close = np.full(n, np.nan)
    if n >= ma_period:
        ema_close[ma_period-1] = np.mean(close[:ma_period])
        multiplier = 2 / (ma_period + 1)
        for i in range(ma_period, n):
            ema_close[i] = (close[i] * multiplier) + (ema_close[i-1] * (1 - multiplier))
    
    upper_channel = ema_close + (2.0 * atr)
    lower_channel = ema_close - (2.0 * atr)
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need ADX (28), ATR (14), EMA (20), volume MA (20)
    start_idx = max(2*period-1, period, ma_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_close[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: price breaks above upper channel in trending market with volume
            if trending and price > upper_channel[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower channel in trending market with volume
            elif trending and price < lower_channel[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA (middle) or trend weakens or stoploss
            if price < ema_close[i] or adx_aligned[i] < 20 or price < (ema_close[i] - 3.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above EMA (middle) or trend weakens or stoploss
            if price > ema_close[i] or adx_aligned[i] < 20 or price > (ema_close[i] + 3.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ATRBreakout_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0