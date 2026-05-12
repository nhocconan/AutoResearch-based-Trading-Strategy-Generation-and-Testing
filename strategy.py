#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R3/S3 levels from weekly timeframe act as strong support/resistance.
# Price breaking above R3 with volume confirmation and weekly trend (price > weekly EMA34) triggers long.
# Price breaking below S3 with volume confirmation and weekly trend (price < weekly EMA34) triggers short.
# Uses weekly timeframe for structure and trend, daily for Camarilla calculation (standard), 12h for execution.
# Designed for 15-25 trades/year with clear breaks and volume to avoid false signals.
# Works in bull via breakouts above resistance and bear via breakdowns below support.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Load daily data for Camarilla calculation (standard practice)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # We use previous day's values to avoid look-ahead
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    
    R3 = prev_close + 1.1 * (prev_high - prev_low)
    S3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema34_val = ema34_1w_aligned[i]
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        vol_confirm = volume_confirm[i]
        close_val = close[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and weekly uptrend (price > weekly EMA34)
            if close_val > R3_val and vol_confirm and close_val > ema34_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and weekly downtrend (price < weekly EMA34)
            elif close_val < S3_val and vol_confirm and close_val < ema34_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (failed breakout/reversal)
            if close_val < S3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (failed breakdown/reversal)
            if close_val > R3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals