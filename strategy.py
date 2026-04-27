#!/usr/bin/env python3
"""
12h Williams Alligator + 1d/21 EMA Trend + Volume Spike + ADX(14) > 20 filter
Williams Alligator: Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3)
Long when Lips > Teeth > Jaw and price > 1d EMA21; Short when Lips < Teeth < Jaw and price < 1d EMA21
Volume > 1.5x average confirms momentum. ADX > 20 ensures trending conditions.
Designed for 12h timeframe with ~15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    for i in range(period, n):
        result[i] = (result[i-1] * (period - 1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA21 and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA(21) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 21
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
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
        minus_dm_period[period-1] = np.sum(minus_dm[1:period+1])
        
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
    
    # Align 1d EMA21 and ADX to 12h timeframe (waits for 1d bar close)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator components on 12h data
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_offset, teeth_offset, lips_offset = 8, 5, 3
    
    # SMMA for median price (HL/2)
    median_price = (high + low) / 2
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply forward shift (offset) - Alligator lines are shifted into the future
    jaw = np.roll(jaw, jaw_offset)
    teeth = np.roll(teeth, teeth_offset)
    lips = np.roll(lips, lips_offset)
    # Set NaN for shifted-out values
    jaw[:jaw_offset] = np.nan
    teeth[:teeth_offset] = np.nan
    lips[:lips_offset] = np.nan
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Alligator (13+8), EMA (21), ADX (28), volume MA (20)
    start_idx = max(lips_period + lips_offset, jaw_period + jaw_offset, ema_period, 2*period-1, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(lips[i]) or
            np.isnan(teeth[i]) or
            np.isnan(jaw[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long entry: bullish alignment + price > 1d EMA21 + trending + volume
            if bullish_alignment and price > ema_1d_aligned[i] and trending and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: bearish alignment + price < 1d EMA21 + trending + volume
            elif bearish_alignment and price < ema_1d_aligned[i] and trending and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish alignment or price < 1d EMA21 or trend weakens
            if bearish_alignment or price < ema_1d_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: bullish alignment or price > 1d EMA21 or trend weakens
            if bullish_alignment or price > ema_1d_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dEMA21_ADX20_Volume"
timeframe = "12h"
leverage = 1.0