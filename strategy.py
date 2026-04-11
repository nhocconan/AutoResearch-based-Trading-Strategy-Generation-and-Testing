#!/usr/bin/env python3
"""
1d_1w_camarilla_breakout
Strategy: 1d breakout with weekly volatility filter and volume confirmation
Timeframe: 1d
Leverage: 1.0
Hypothesis: Buy when 1d closes above prior week's R3 with volume expansion and low volatility regime; sell when 1d closes below prior week's S3 with same conditions. Uses weekly ATR-based volatility filter to avoid choppy markets and volume expansion to confirm breakouts. Designed for both bull and bear markets by focusing on volatility breakouts rather than trend direction, which works in ranging and trending conditions. Low-frequency design targets 10-20 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly ATR (volatility filter: low volatility regime) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR
    tr1_1w = high_1w[1:] - low_1w[1:]
    tr2_1w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_1w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly ATR ratio: current ATR / 20-period average ATR (low when < 0.8)
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1w / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # === Weekly Close (prior close for context) ===
    close_1w_shifted = np.roll(close_1w, 1)
    close_1w_shifted[0] = np.nan
    close_1w_prior = align_htf_to_ltf(prices, df_1w, close_1w_shifted)
    
    # === Weekly Camarilla (entry levels from prior week) ===
    high_1w_shift = np.roll(high_1w, 1)
    low_1w_shift = np.roll(low_1w, 1)
    close_1w_shift = np.roll(close_1w, 1)
    high_1w_shift[0] = np.nan
    low_1w_shift[0] = np.nan
    close_1w_shift[0] = np.nan
    
    pivot_1w = (high_1w_shift + low_1w_shift + close_1w_shift) / 3
    range_1w = high_1w_shift - low_1w_shift
    r3_1w = close_1w_shift + range_1w * 1.166
    s3_1w = close_1w_shift - range_1w * 1.166
    
    # Align weekly Camarilla to 1d timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(close_1w_prior[i]) or np.isnan(atr_ratio_aligned[i]) or np.isnan(atr_1d[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 1d volume must be expanded
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Volatility filter: low volatility regime (ATR ratio < 0.8)
        low_volatility = atr_ratio_aligned[i] < 0.8
        
        # Long conditions: 1d closes above prior week's R3 with volume expansion + low volatility
        long_signal = volume_expanded and low_volatility and (price_close > r3_1w_aligned[i])
        
        # Short conditions: 1d closes below prior week's S3 with volume expansion + low volatility
        short_signal = volume_expanded and low_volatility and (price_close < s3_1w_aligned[i])
        
        # Exit when price returns to the weekly pivot (mean reversion within prior week's range)
        exit_long = position == 1 and price_close < pivot_1w_aligned[i]
        exit_short = position == -1 and price_close > pivot_1w_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Buy when 1d closes above prior week's R3 with volume expansion and low volatility regime; sell when 1d closes below prior week's S3 with same conditions. Uses weekly ATR-based volatility filter to avoid choppy markets and volume expansion to confirm breakouts. Designed for both bull and bear markets by focusing on volatility breakouts rather than trend direction, which works in ranging and trending conditions. Low-frequency design targets 10-20 trades/year to minimize fee drift.