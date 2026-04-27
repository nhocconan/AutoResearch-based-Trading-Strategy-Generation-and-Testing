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
    
    # Get 12h data for calculations (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] * 2 + ema_34_12h[i-1] * 32) / 34  # EMA34
    
    # Calculate 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                               np.abs(low_12h[1:] - close_12h[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_12h = np.full(len(close_12h), np.nan)
    for i in range(14, len(close_12h)):
        if i == 14:
            atr_12h[i] = np.mean(tr[1:15])
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Calculate 12h ADX for trend strength
    # +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, TR
    tr_12h = tr  # Already calculated above
    period = 14
    tr_sum = np.full(len(tr_12h), np.nan)
    plus_dm_sum = np.full(len(plus_dm), np.nan)
    minus_dm_sum = np.full(len(minus_dm), np.nan)
    
    if len(tr_12h) >= period:
        tr_sum[period-1] = np.sum(tr_12h[1:period+1])
        plus_dm_sum[period-1] = np.sum(plus_dm[1:period+1])
        minus_dm_sum[period-1] = np.sum(minus_dm[1:period+1])
        for i in range(period, len(tr_12h)):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr_12h[i]
            plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
            minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = np.full(len(tr_12h), np.nan)
    minus_di = np.full(len(tr_12h), np.nan)
    dx = np.full(len(tr_12h), np.nan)
    for i in range(period, len(tr_12h)):
        if tr_sum[i] != 0:
            plus_di[i] = 100 * (plus_dm_sum[i] / tr_sum[i])
            minus_di[i] = 100 * (minus_dm_sum[i] / tr_sum[i])
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.full(len(tr_12h), np.nan)
    if len(dx) >= period:
        valid_dx = dx[period-1:]
        if len(valid_dx) >= period:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 12h indicators to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6-period volume average for volume filter
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period - ensure all indicators are valid
    start_idx = max(34, vol_period, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # Trend strength filter: ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Price above EMA34 with volume and trend strength
            if price > ema_34_12h_aligned[i] and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA34 with volume and trend strength
            elif price < ema_34_12h_aligned[i] and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA34 or volatility drops
            if price < ema_34_12h_aligned[i] or atr_12h_aligned[i] < (atr_12h_aligned[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA34 or volatility drops
            if price > ema_34_12h_aligned[i] or atr_12h_aligned[i] < (atr_12h_aligned[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_EMA34_ADX_Volume_Filter"
timeframe = "6h"
leverage = 1.0