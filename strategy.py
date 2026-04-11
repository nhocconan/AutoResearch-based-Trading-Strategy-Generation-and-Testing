#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_atr_volatility_v4
Strategy: 4h breakout with 1d ATR volatility filter and volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 1d ATR-based volatility filter (ATR < 20-period average) combined with volume expansion and Camarilla breakouts. Designed to capture volatility breakouts in both bull and bear markets with low trade frequency to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_atr_volatility_v4"
timeframe = "4h"
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
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h ATR for stoploss (not used in signal, but kept for potential use)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d ATR calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR ratio: current ATR / 20-period average
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === 1d Close (prior close for context) ===
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_prior = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 1d Camarilla (from prior day) ===
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166
    s3_1d = close_1d_shift - range_1d * 1.166
    
    # Align 1d Camarilla to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(close_1d_prior[i]) or np.isnan(atr_ratio_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume expansion: 4h volume > 2.0x 20-period average
        volume_expanded = volume_current > 2.0 * vol_ma
        
        # Volatility filter: low volatility regime (ATR ratio < 0.8)
        low_volatility = atr_ratio_aligned[i] < 0.8
        
        # Strong candle confirmation
        strong_bullish = price_close > price_open
        strong_bearish = price_close < price_open
        
        # Long: break above R3 with volume expansion, low volatility, and bullish candle
        long_signal = volume_expanded and low_volatility and strong_bullish and (price_close > r3_1d_aligned[i])
        
        # Short: break below S3 with volume expansion, low volatility, and bearish candle
        short_signal = volume_expanded and low_volatility and strong_bearish and (price_close < s3_1d_aligned[i])
        
        # Exit when price returns to the 1d pivot (mean reversion)
        exit_long = position == 1 and price_close < pivot_1d_aligned[i]
        exit_short = position == -1 and price_close > pivot_1d_aligned[i]
        
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

# Hypothesis: Uses 1d ATR-based volatility filter (ATR < 20-period average) combined with volume expansion and Camarilla breakouts. 
# Designed to capture volatility breakouts in both bull and bear markets with low trade frequency to avoid fee drift.