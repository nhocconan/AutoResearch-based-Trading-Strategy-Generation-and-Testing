#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_trix_volume_regime_v2"
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
    
    # Get 1d data for TRIX and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate TRIX(12) on daily data: triple EMA of % change
    roc = np.diff(close_1d) / close_1d[:-1]
    roc = np.concatenate([[np.nan], roc])  # align with original length
    
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values * 100  # scale for readability
    
    # Align TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume filter: current 6h volume > 20-period average of 1d volume
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_ok = volume > vol_ma_1d_aligned
    
    # Trend filter: price above/below 50-period EMA on 6h
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX signals with volume confirmation
        # Long: TRIX rising (>0) in uptrend with volume
        long_signal = trix_aligned[i] > 0 and uptrend[i] and volume_ok[i]
        # Short: TRIX falling (<0) in downtrend with volume
        short_signal = trix_aligned[i] < 0 and downtrend[i] and volume_ok[i]
        
        # Exit when TRIX reverses
        exit_long = trix_aligned[i] < 0
        exit_short = trix_aligned[i] > 0
        
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