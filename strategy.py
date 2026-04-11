#!/usr/bin/env python3
"""
4h_12h_1d_camarilla_pullback_v1
Strategy: 4h pullback to 12h/1d confluence levels with volume and volatility filters
Timeframe: 4h
Leverage: 1.0
Hypothesis: Combines 12h and 1d Camarilla levels to identify high-probability pullback entries. 
Long when price pulls back to 12h/1d S3/S4 in a bullish 1d trend; short when price pulls back to 12h/1d R3/R4 in a bearish 1d trend.
Requires volume > 1.8x 20-period average and ATR ratio < 0.7 (low volatility regime) to avoid chop.
Designed for fewer, higher-quality trades (target: 20-35/year) to minimize fee drag and improve generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_camarilla_pullback_v1"
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h ATR (volatility filter) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    atr_ma_20_12h = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    atr_ratio_12h = atr_12h / atr_ma_20_12h
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    
    # === 1d ATR (volatility filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_20_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # === 12h Close (prior close for trend) ===
    close_12h_shifted = np.roll(close_12h, 1)
    close_12h_shifted[0] = np.nan
    close_12h_prior = align_htf_to_ltf(prices, df_12h, close_12h_shifted)
    
    # === 1d Close (prior close for trend) ===
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_prior = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 12h Camarilla (from prior 12h) ===
    high_12h_shift = np.roll(high_12h, 1)
    low_12h_shift = np.roll(low_12h, 1)
    close_12h_shift = np.roll(close_12h, 1)
    high_12h_shift[0] = np.nan
    low_12h_shift[0] = np.nan
    close_12h_shift[0] = np.nan
    
    pivot_12h = (high_12h_shift + low_12h_shift + close_12h_shift) / 3
    range_12h = high_12h_shift - low_12h_shift
    r3_12h = close_12h_shift + range_12h * 1.166
    r4_12h = close_12h_shift + range_12h * 1.500
    s3_12h = close_12h_shift - range_12h * 1.166
    s4_12h = close_12h_shift - range_12h * 1.500
    
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 1d Camarilla (from prior 1d) ===
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166
    r4_1d = close_1d_shift + range_1d * 1.500
    s3_1d = close_1d_shift - range_1d * 1.166
    s4_1d = close_1d_shift - range_1d * 1.500
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(close_12h_prior[i]) or np.isnan(close_1d_prior[i]) or
            np.isnan(atr_ratio_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 4h volume must be expanded
        volume_expanded = volume_current > 1.8 * vol_ma
        
        # Volatility filter: low volatility regime on both 12h and 1d
        low_volatility = (atr_ratio_12h_aligned[i] < 0.7) and (atr_ratio_1d_aligned[i] < 0.7)
        
        # Determine 1d trend (bullish/bearish)
        bullish_1d = close_1d_prior[i] > close_1d[i]  # Prior close > current close = uptrend
        bearish_1d = close_1d_prior[i] < close_1d[i]  # Prior close < current close = downtrend
        
        # Long conditions: pullback to 12h/1d support in bullish 1d trend
        near_12h_support = (price_close <= s3_12h_aligned[i] * 1.002) and (price_close >= s4_12h_aligned[i] * 0.998)
        near_1d_support = (price_close <= s3_1d_aligned[i] * 1.002) and (price_close >= s4_1d_aligned[i] * 0.998)
        long_signal = volume_expanded and low_volatility and bullish_1d and (near_12h_support or near_1d_support)
        
        # Short conditions: pullback to 12h/1d resistance in bearish 1d trend
        near_12h_resistance = (price_close >= r3_12h_aligned[i] * 0.998) and (price_close <= r4_12h_aligned[i] * 1.002)
        near_1d_resistance = (price_close >= r3_1d_aligned[i] * 0.998) and (price_close <= r4_1d_aligned[i] * 1.002)
        short_signal = volume_expanded and low_volatility and bearish_1d and (near_12h_resistance or near_1d_resistance)
        
        # Exit when price returns to the 12h/1d pivot (mean reversion)
        exit_long = position == 1 and price_close > ((pivot_12h_aligned[i] if not np.isnan(pivot_12h_aligned[i]) else pivot_1d_aligned[i]) * 0.998)
        exit_short = position == -1 and price_close < ((pivot_12h_aligned[i] if not np.isnan(pivot_12h_aligned[i]) else pivot_1d_aligned[i]) * 1.002)
        
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