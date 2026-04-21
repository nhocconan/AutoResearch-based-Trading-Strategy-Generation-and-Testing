#!/usr/bin/env python3
"""
1d_1w_12h_Camarilla_R1S1_Breakout_Volume_Tight_v1
Hypothesis: Use 1w for trend filter, 12h for regime (ADX), and 1d for entry with 1d R1/S1 breakouts.
Long when price breaks above 1d R1, weekly trend up (weekly close > weekly open), and 12h ADX > 25.
Short when price breaks below 1d S1, weekly trend down (weekly close < weekly open), and 12h ADX > 25.
Exit when price crosses 1d pivot point. Volume confirmation (1.5x 20-period avg) reduces false breaks.
Target: 15-25 trades/year per symbol. Works in bull/bear by following weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for entry levels (R1, S1, PP)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to daily timeframe (no shift needed as these are based on previous day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_trend_up = weekly_close > weekly_open
    weekly_trend_down = weekly_close < weekly_open
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Load 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                               np.abs(low_12h[1:] - close_12h[:-1])))
    # Pad to match length
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    tr = np.concatenate([[np.nan], tr])
    
    # Smoothed values
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period] = np.nanmean(tr[1:atr_period+1])
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    plus_di = 100 * (np.zeros_like(plus_dm))
    minus_di = 100 * (np.zeros_like(minus_dm))
    for i in range(atr_period, len(plus_dm)):
        if atr[i] != 0 and not np.isnan(atr[i]):
            plus_di[i] = 100 * np.nansum(plus_dm[i-atr_period+1:i+1]) / (atr[i] * atr_period)
            minus_di[i] = 100 * np.nansum(minus_dm[i-atr_period+1:i+1]) / (atr[i] * atr_period)
    
    dx = np.zeros_like(plus_di)
    for i in range(len(dx)):
        if (plus_di[i] + minus_di[i]) != 0 and not (np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros_like(dx)
    adx[2*atr_period-1] = np.nanmean(dx[atr_period:2*atr_period])
    for i in range(2*atr_period, len(adx)):
        if not np.isnan(dx[i]):
            adx[i] = (adx[i-1] * (atr_period-1) + dx[i]) / atr_period
        else:
            adx[i] = adx[i-1]
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter
        weekly_up = weekly_trend_up_aligned[i]
        weekly_down = weekly_trend_down_aligned[i]
        
        # ADX filter: trending market (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long conditions: break above R1 + volume + weekly uptrend + strong trend
            if price > r1_aligned[i] and volume_ok and weekly_up and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume + weekly downtrend + strong trend
            elif price < s1_aligned[i] and volume_ok and weekly_down and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_12h_Camarilla_R1S1_Breakout_Volume_Tight_v1"
timeframe = "1d"
leverage = 1.0