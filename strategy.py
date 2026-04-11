#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_cci_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate Weekly CCI for trend filter (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price
    tp_1w = (high_1w + low_1w + close_1w) / 3
    # 20-period SMA of typical price
    sma_tp_20 = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md_20 = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI calculation: (TP - SMA) / (0.015 * MD)
    cci_20 = (tp_1w - sma_tp_20) / (0.015 * md_20)
    # Replace inf/NaN from zero MD
    cci_20 = np.where(np.isnan(cci_20) | np.isinf(cci_20), 0, cci_20)
    
    # Align weekly CCI to 6h
    cci_20_aligned = align_htf_to_ltf(prices, df_1w, cci_20)
    
    # Calculate Daily CCI for entry timing (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3
    # 14-period SMA of typical price
    sma_tp_14 = pd.Series(tp_1d).rolling(window=14, min_periods=14).mean().values
    # Mean deviation
    md_14 = pd.Series(tp_1d).rolling(window=14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI calculation
    cci_14 = (tp_1d - sma_tp_14) / (0.015 * md_14)
    # Replace inf/NaN from zero MD
    cci_14 = np.where(np.isnan(cci_14) | np.isinf(cci_14), 0, cci_14)
    
    # Align daily CCI to 6h
    cci_14_aligned = align_htf_to_ltf(prices, df_1d, cci_14)
    
    # Volume confirmation: 6h volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_20_aligned[i]) or np.isnan(cci_14_aligned[i]) or 
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Weekly CCI trend filter: > 100 for uptrend, < -100 for downtrend
        weekly_uptrend = cci_20_aligned[i] > 100
        weekly_downtrend = cci_20_aligned[i] < -100
        
        # Daily CCI for entry: oversold/overbought conditions
        daily_oversold = cci_14_aligned[i] < -100
        daily_overbought = cci_14_aligned[i] > 100
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_50[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Weekly uptrend + Daily oversold + Volume confirmation
        if weekly_uptrend and daily_oversold and vol_confirm:
            enter_long = True
        
        # Short: Weekly downtrend + Daily overbought + Volume confirmation
        if weekly_downtrend and daily_overbought and vol_confirm:
            enter_short = True
        
        # Exit conditions: CCI crosses back through zero
        exit_long = cci_14_aligned[i] > 0  # Exit long when CCI turns positive
        exit_short = cci_14_aligned[i] < 0  # Exit short when CCI turns negative
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h CCI breakout strategy using weekly CCI for trend filter and daily CCI for entry timing.
# Weekly CCI (>100 for uptrend, <-100 for downtrend) determines the primary trend direction from 1w data.
# Daily CCI (<-100 for long entry, >100 for short entry) provides oversold/overbought signals for entry timing.
# Volume confirmation (>1.5x 50-period average) ensures institutional participation.
# Exits when daily CCI crosses back through zero, capturing mean reversion within the trend.
# Designed to work in both bull and bear markets by following the weekly trend while fading daily extremes.
# Target: 20-50 trades per year (80-200 total over 4 years) to minimize fee drag.
# Position size: 0.25 (25% of capital) to manage risk during volatile periods.
# Uses 6h timeframe for balance between signal quality and trade frequency.