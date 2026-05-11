#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe with 1d EMA trend filter and volume confirmation.
Long when price breaks above R3 with 1d uptrend and volume confirmation.
Short when price breaks below S3 with 1d downtrend and volume confirmation.
Exit when price retests the pivot level (PP) or trend reverses.
Camarilla levels provide high-probability reversal/breakout zones. Works in bull by catching breakouts,
in bear by fading reversals at key levels. Volume filter ensures participation. Targets ~25-40 trades/year.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels from 1d (OHLC of previous day) ---
    # Calculate for each 4h bar using prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    PP = (high_1d + low_1d + close_1d) / 3  # Pivot Point
    
    # Align to 4h timeframe (values available after 1d bar closes)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    PP_4h = align_htf_to_ltf(prices, df_1d, PP)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(PP_4h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for breakouts in direction of 1d trend with volume
            if close_4h[i] > R3_4h[i] and trend_up and vol_ok:
                # Break long: price above R3 with uptrend and volume
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < S3_4h[i] and trend_down and vol_ok:
                # Break short: price below S3 with downtrend and volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price retests PP OR trend turns down
                if close_4h[i] <= PP_4h[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price retests PP OR trend turns up
                if close_4h[i] >= PP_4h[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals