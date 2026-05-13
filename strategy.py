#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with daily trend filter and volume confirmation.
# Uses daily EMA50 to determine trend direction, then enters long when price breaks above R1 in uptrend,
# or short when breaks below S1 in downtrend. Volume > 20-period average confirms breakout strength.
# Exits when price returns to pivot point (daily PP). Designed for 20-35 trades/year to minimize fee drag.
# Camarilla levels provide precise intraday support/resistance, proven effective in both bull/bear markets.

name = "4h_Camarilla_R1_S1_Breakout_1D_Trend_Force_v2"
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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d_arr, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Camarilla calculations
    range_1d = phigh - plow
    # R1 = pclose + range_1d * 1.1/12
    # S1 = pclose - range_1d * 1.1/12
    # PP = (phigh + plow + pclose) / 3
    r1 = pclose + range_1d * 1.1 / 12
    s1 = pclose - range_1d * 1.1 / 12
    pp = (phigh + plow + pclose) / 3
    
    # Align daily levels to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA50
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 in uptrend (price > EMA50) with volume
            if close[i] > r1_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 in downtrend (price < EMA50) with volume
            elif close[i] < s1_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to or below pivot point
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to or above pivot point
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals