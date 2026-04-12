#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX on daily close (15-period triple EMA)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix_raw.values
    
    # Calculate 34-period EMA for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1w data for regime filter (choppiness-like using range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Weekly range as volatility measure
    weekly_range = high_1w - low_1w
    # Use 8-week average range for normalization
    avg_range = pd.Series(weekly_range).rolling(window=8, min_periods=8).mean().values
    # Avoid division by zero
    range_ratio = np.where(avg_range > 0, weekly_range / avg_range, 1.0)
    
    # Align all indicators to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    range_ratio_aligned = align_htf_to_ltf(prices, df_1w, range_ratio)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(range_ratio_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX signal: zero line cross
        trix_pos = trix_aligned[i] > 0
        trix_neg = trix_aligned[i] < 0
        
        # Trend filter: price vs 34 EMA
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Regime filter: avoid extreme volatility (range_ratio > 2.0 = choppy)
        low_volatility = range_ratio_aligned[i] < 2.0
        
        # Entry conditions with volume confirmation
        # Long: TRIX positive AND uptrend AND low volatility AND volume
        long_signal = trix_pos and uptrend and low_volatility and volume_ok[i]
        # Short: TRIX negative AND downtrend AND low volatility AND volume
        short_signal = trix_neg and downtrend and low_volatility and volume_ok[i]
        
        # Exit when TRIX crosses zero
        exit_long = trix_aligned[i] <= 0
        exit_short = trix_aligned[i] >= 0
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals