#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout on 4h with 1-day EMA34 trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal/breakout points based on prior day's range.
# The 1-day EMA34 filter ensures alignment with daily trend, reducing counter-trend trades.
# Volume confirmation ensures breakouts have institutional backing. Designed to work in both bull and bear markets
# by following the higher timeframe trend while capturing intraday momentum at key levels.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-day Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1-day for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla Levels from Previous Day ===
    # Calculate daily high, low, close
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    # Shift to get previous day's values for current day's calculation
    phigh = np.concatenate([[np.nan], high_1d[:-1]])
    plow = np.concatenate([[np.nan], low_1d[:-1]])
    pclose = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    range_ = phigh - plow
    r1 = pclose + (range_ * 1.1 / 12)
    s1 = pclose - (range_ * 1.1 / 12)
    r2 = pclose + (range_ * 1.1 / 6)
    s2 = pclose - (range_ * 1.1 / 6)
    r3 = pclose + (range_ * 1.1 / 4)
    s3 = pclose - (range_ * 1.1 / 4)
    r4 = pclose + (range_ * 1.1 / 2)
    s4 = pclose - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s4)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1-day EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and daily uptrend
            if (close[i] > r1_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and daily downtrend
            elif (close[i] < s1_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below R1 or daily trend changes
            if (close[i] < r1_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above S1 or daily trend changes
            if (close[i] > s1_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals