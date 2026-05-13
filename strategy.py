#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1D_Trend_Force_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1D EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily Camarilla levels (based on previous day)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    camarilla_R1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_S1[i] = prev_close - 1.1 * range_val / 12
        camarilla_R1[i] = prev_close + 1.1 * range_val / 12
    
    # Align 1D indicators to 4H timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    
    # Volume confirmation: current volume vs 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 + uptrend + volume confirmation
            if close[i] > camarilla_R1_aligned[i] and price_above_ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + downtrend + volume confirmation
            elif close[i] < camarilla_S1_aligned[i] and price_below_ema and volume_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend changes
            if close[i] < camarilla_S1_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend changes
            if close[i] > camarilla_R1_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals