#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_Filter
# Hypothesis: Fade price reversals at Camarilla R3/S3 levels during low volatility (ADX < 20) in the direction of 1d EMA34 trend, with volume confirmation.
# Breakouts above R4 or below S4 with volume spike and trend alignment continue the move.
# Works in bull (buy dips to S3 in uptrend) and bear (sell rallies to R3 in downtrend).
# Low frequency due to multi-condition confluence (level, volatility, trend, volume).

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # Using previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # fill first value
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    daily_range = prev_high - prev_low
    R3 = prev_close + 1.1 * daily_range
    S3 = prev_close - 1.1 * daily_range
    R4 = prev_close + 1.5 * daily_range
    S4 = prev_close - 1.5 * daily_range
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # ADX for volatility regime (14-period)
    # +DM, -DM, TR
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align daily indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: volume > 1.8 * 4-period average
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 1.8 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # FADE at R3/S3 in low volatility (ADX < 20) with trend alignment
            # LONG: Price near S3, ADX < 20, price > EMA34 (uptrend), volume spike
            if (close[i] <= S3_aligned[i] * 1.002 and  # within 0.2% of S3
                adx_aligned[i] < 20 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near R3, ADX < 20, price < EMA34 (downtrend), volume spike
            elif (close[i] >= R3_aligned[i] * 0.998 and  # within 0.2% of R3
                  adx_aligned[i] < 20 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            # BREAKOUT: Continue trend at R4/S4 with volume spike
            # LONG breakout: price > R4, uptrend, volume spike
            elif (close[i] > R4_aligned[i] and 
                  close[i] > ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT breakdown: price < S4, downtrend, volume spike
            elif (close[i] < S4_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R4 or trend breaks
            if close[i] >= R4_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S4 or trend breaks
            if close[i] <= S4_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals