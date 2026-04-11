#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance and support levels (previous day's data)
    r3 = close_1d + range_1d * 1.166
    s3 = close_1d - range_1d * 1.166
    r4 = close_1d + range_1d * 1.500
    s4 = close_1d - range_1d * 1.500
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r4 = np.roll(r4, 1)
    s4 = np.roll(s4, 1)
    r3[0] = np.nan
    s3[0] = np.nan
    r4[0] = np.nan
    s4[0] = np.nan
    
    # Align daily Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly ADX for trend strength (14 period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    tr_dm_1w = tr_1w[1:]
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm_1w).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm_1w).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to 6h timeframe
    adx_1w_6h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 6h ATR for volatility filter (14 period)
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(adx_1w_6h[i]) or np.isnan(atr_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Weekly trend filter: ADX > 25 (strong trend filter to reduce trades)
        trend_filter = adx_1w_6h[i] > 25
        
        # Long conditions: price breaks above R4 with volume and weekly trend
        long_signal = volume_confirmed and trend_filter and (price_high > r4_6h[i])
        
        # Short conditions: price breaks below S4 with volume and weekly trend
        short_signal = volume_confirmed and trend_filter and (price_low < s4_6h[i])
        
        # Exit when price returns to the weekly pivot level (mean reversion)
        # Calculate weekly pivot from weekly data
        pivot_1w = (high_1w + low_1w + close_1w) / 3
        pivot_1w_shifted = np.roll(pivot_1w, 1)
        pivot_1w_shifted[0] = np.nan
        pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w_shifted)
        
        exit_long = position == 1 and price_close < pivot_1w_6h[i]
        exit_short = position == -1 and price_close > pivot_1w_6h[i]
        
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

# Hypothesis: 6h Camarilla breakout strategy using weekly ADX for trend filter and daily R4/S4 levels for breakout signals.
# Enters long when 6h price breaks above daily R4 level (close + 1.500*range) with volume >1.5x average and weekly ADX>25.
# Enters short when price breaks below daily S4 level (close - 1.500*range) with same conditions.
# Exits when price returns to the weekly pivot level (mean reversion within the week's range).
# Uses R4/S4 levels (extreme levels) to reduce false breakouts and increase win rate in trending markets.
# Weekly ADX filter ensures we only trade in strong weekly trends, avoiding choppy markets.
# Volume confirmation adds conviction to breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear markets as it adapts to weekly volatility ranges and trends.