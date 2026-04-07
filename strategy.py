#!/usr/bin/env python3
"""
4h_1d_ema_breakout_volume_v1
Hypothesis: On 4h timeframe, enter long when price breaks above 1d EMA50 with above-average volume and short when price breaks below 1d EMA50 with above-average volume. Use 4h ATR to filter for low volatility environments where breakouts are more likely to succeed. Exit when price crosses back below/above EMA or volatility increases significantly. This strategy captures trend continuation in both bull and bear markets by following the higher timeframe trend with volume confirmation. Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h ATR for volatility filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile rank (50-day lookback for volatility regime)
    atr_percentile = pd.Series(atr_4h).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr_percentile[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR percentile below 60th percentile
        low_vol = atr_percentile[i] < 0.6
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Breakout conditions
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses back below EMA or volatility increases significantly
            if close[i] < ema_1d_aligned[i] or atr_percentile[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above EMA or volatility increases significantly
            if close[i] > ema_1d_aligned[i] or atr_percentile[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol and vol_ok:
                # Breakout above EMA with volume - go long
                if above_ema and close[i] > ema_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Breakout below EMA with volume - go short
                elif below_ema and close[i] < ema_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals