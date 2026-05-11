#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use 12h timeframe with Camarilla pivot levels (R3/S3) from 1d data. Go long when price breaks above R3 in 1d uptrend with volume confirmation, short when breaks below S3 in 1d downtrend with volume confirmation. Uses 1w trend filter for regime alignment. Designed for low trade frequency (12-37/year) to minimize fee drift. Works in bull via R3 breakouts in uptrend, bear via S3 breakdowns in downtrend.

name = "12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivots (R3, S3) ---
    # Based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    R3 = np.zeros_like(high_1d)
    S3 = np.zeros_like(low_1d)
    for i in range(len(close_1d)):
        if i == 0:
            R3[i] = high_1d[i]
            S3[i] = low_1d[i]
        else:
            range_ = high_1d[i-1] - low_1d[i-1]
            close_prev = close_1d[i-1]
            R3[i] = close_prev + 1.1 * range_ / 6
            S3[i] = close_prev - 1.1 * range_ / 6
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    R3_prev = np.roll(R3, 1)
    S3_prev = np.roll(S3, 1)
    R3_prev[0] = R3[0]
    S3_prev[0] = S3[0]
    
    # Align to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_prev)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_prev)
    
    # --- 1d Trend: EMA34 slope ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_34_1d = np.diff(ema_34_1d, prepend=ema_34_1d[0])
    ema_slope_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_34_1d)
    
    # --- 1w Trend Filter: EMA20 slope ---
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slope_20_1w = np.diff(ema_20_1w, prepend=ema_20_1w[0])
    ema_slope_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_20_1w)
    
    # --- Volatility (ATR) for stop ---
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Volume Confirmation: 2.0x 20-period average ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(ema_slope_34_1d_aligned[i]) or
            np.isnan(ema_slope_20_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filters
        bullish_1d = ema_slope_34_1d_aligned[i] > 0
        bearish_1d = ema_slope_34_1d_aligned[i] < 0
        bullish_1w = ema_slope_20_1w_aligned[i] > 0
        bearish_1w = ema_slope_20_1w_aligned[i] < 0
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 in 1d uptrend AND 1w uptrend with volume surge
            if close[i] > R3_aligned[i] and bullish_1d and bullish_1w and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below S3 in 1d downtrend AND 1w downtrend with volume surge
            elif close[i] < S3_aligned[i] and bearish_1d and bearish_1w and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 2.5*ATR from highest high
                if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 2.5*ATR from lowest low
                if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals