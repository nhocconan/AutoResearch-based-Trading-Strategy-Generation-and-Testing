#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1D_Trend_Force_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1D data ONCE for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for previous day
    def calculate_camarilla(high, low, close):
        """Calculate Camarilla pivot levels: R1, R2, S1, S2"""
        range_ = high - low
        c = close
        r1 = c + (range_ * 1.1 / 12)
        r2 = c + (range_ * 1.1 / 6)
        s1 = c - (range_ * 1.1 / 12)
        s2 = c - (range_ * 1.1 / 6)
        return r1, r2, s1, s2
    
    # Calculate for each day (using previous day's data)
    r1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        r1[i], r2[i], s1[i], s2[i] = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
    
    # Calculate 1D EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1D volume average for volume spike filter
    vol_1d_series = pd.Series(volume_1d)
    vol_avg_20 = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all 1D indicators to 4H timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4H volume > 1.5x 20-day average volume
        vol_spike = volume[i] > (1.5 * vol_avg_20_aligned[i])
        
        # Trend filter: price above/below EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume + uptrend
            if close[i] > r1_aligned[i] and vol_spike and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume + downtrend
            elif close[i] < s1_aligned[i] and vol_spike and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend weakens
            if close[i] < s1_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend weakens
            if close[i] > r1_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals