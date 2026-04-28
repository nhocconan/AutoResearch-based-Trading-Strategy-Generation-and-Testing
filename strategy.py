#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with 1d EMA34 trend filter and volume confirmation.
# In ranging markets (price between R3-S3): fade extremes at R3/S3 with volume confirmation.
# In trending markets (price > R4 or < S4): breakout continuation in direction of 1d EMA34 trend.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-150 trades over 4 years.
# Camarilla levels provide mathematically derived support/resistance, volume confirms momentum, 1d EMA34 filters counter-trend noise.
# Works in bull (breakouts with trend) and bear (mean reversion at extremes) markets.

name = "6h_WeeklyCamarilla_R3S3_R4S4_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Initialize arrays for Camarilla levels (same length as weekly data)
    R4 = np.full_like(close_1w, np.nan)
    R3 = np.full_like(close_1w, np.nan)
    S3 = np.full_like(close_1w, np.nan)
    S4 = np.full_like(close_1w, np.nan)
    PP = np.full_like(close_1w, np.nan)
    
    # Calculate Camarilla levels for each completed weekly bar
    for i in range(len(close_1w)):
        H = high_1w[i]
        L = low_1w[i]
        C = close_1w[i]
        PP[i] = (H + L + C) / 3
        R4[i] = C + ((H - L) * 1.1 / 2)
        R3[i] = C + ((H - L) * 1.1 / 4)
        S3[i] = C - ((H - L) * 1.1 / 4)
        S4[i] = C - ((H - L) * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe (with 1-bar delay for completed weekly bar)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    
    # Get daily data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Market regime detection
        # Ranging: price between R3 and S3
        # Trending: price > R4 (bullish) or < S4 (bearish)
        is_ranging = (close[i] >= S3_aligned[i]) and (close[i] <= R3_aligned[i])
        is_bullish_trend = close[i] > R4_aligned[i]
        is_bearish_trend = close[i] < S4_aligned[i]
        
        # Mean reversion signals (ranging market)
        long_mean_reversion = (is_ranging and 
                              close[i] <= S3_aligned[i] and  # at or below S3
                              volume_confirm[i] and
                              close[i] > ema_34_1d_aligned[i])  # only long if above daily EMA (bullish bias)
        
        short_mean_reversion = (is_ranging and
                               close[i] >= R3_aligned[i] and  # at or above R3
                               volume_confirm[i] and
                               close[i] < ema_34_1d_aligned[i])  # only short if below daily EMA (bearish bias)
        
        # Breakout continuation signals (trending market)
        long_breakout = (is_bullish_trend and
                        close[i] > ema_34_1d_aligned[i] and  # only long if above daily EMA
                        volume_confirm[i])
        
        short_breakout = (is_bearish_trend and
                         close[i] < ema_34_1d_aligned[i] and  # only short if below daily EMA
                         volume_confirm[i])
        
        # Exit conditions
        # Exit long: price reaches R3 (mean reversion target) or breaks below S4 (trend failure)
        long_exit = (close[i] >= R3_aligned[i]) or (close[i] < S4_aligned[i])
        
        # Exit short: price reaches S3 (mean reversion target) or breaks above R4 (trend failure)
        short_exit = (close[i] <= S3_aligned[i]) or (close[i] > R4_aligned[i])
        
        # Handle entries and exits
        if (long_mean_reversion or long_breakout) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_mean_reversion or short_breakout) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals