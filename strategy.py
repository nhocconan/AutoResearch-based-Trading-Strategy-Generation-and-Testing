#!/usr/bin/env python3
"""
12H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_FILTER
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d trend filter and volume confirmation. 
Breakouts only when price closes above R1 or below S1, EMA50 trend aligns with breakout, 
and volume > 1.5x 20-day average. This filters out low-conviction breakouts and focuses on 
institutional participation. Designed for 12h timeframe with fewer, higher-quality trades 
to avoid fee drag and work in both bull and bear markets.
"""
name = "12H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_FILTER"
timeframe = "12h"
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
    
    # Camarilla levels from previous 12h bar (using typical price)
    typical_price = (high + low + close) / 3
    # Shift by 1 to use previous bar's data
    prev_typical = np.roll(typical_price, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_typical[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels (R1, S1)
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 1d data for trend filter (EMA50) and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current day volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = vol_1d / vol_ma_20d
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to avoid NaN from roll
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close above R1 with uptrend and volume confirmation
            if (close[i] > R1[i] and 
                close[i] > ema50_aligned[i] and 
                vol_ratio_aligned[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 with downtrend and volume confirmation
            elif (close[i] < S1[i] and 
                  close[i] < ema50_aligned[i] and 
                  vol_ratio_aligned[i] > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal
            if close[i] < S1[i] or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal
            if close[i] > R1[i] or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals