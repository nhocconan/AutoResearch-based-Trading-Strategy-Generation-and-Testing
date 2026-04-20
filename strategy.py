#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) on 1d data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Middle band: 20-period SMA
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Calculate Bollinger Band Width for volatility filter
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate ADX (14) on 1d data for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = bb_upper_aligned[i]
        lower_val = bb_lower_aligned[i]
        sma_val = sma_20_aligned[i]
        bb_width_val = bb_width_aligned[i]
        adx_val = adx_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(sma_val) or np.isnan(bb_width_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when BB width is above 20th percentile (avoid chop)
        # Calculate percentile using rolling window of last 50 values
        if i >= 50:
            bb_width_history = bb_width_aligned[max(0, i-49):i+1]
            bb_width_percentile = (bb_width_val > bb_width_history).sum() / len(bb_width_history) * 100
        else:
            bb_width_percentile = 50  # neutral
        
        # Only trade when volatility is sufficient (above 30th percentile)
        volatility_filter = bb_width_percentile > 30
        
        if position == 0:
            # Long: price touches lower BB and ADX shows weak trend (range-bound)
            if close_val <= lower_val and adx_val < 25 and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB and ADX shows weak trend (range-bound)
            elif close_val >= upper_val and adx_val < 25 and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches middle BB or stops trending
            if close_val >= sma_val or adx_val > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches middle BB or stops trending
            if close_val <= sma_val or adx_val > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_BollingerRange_ADXFilter_V1
# Uses 1-day Bollinger Bands (20,2) as mean reversion boundaries
# Enters long when 12h price touches lower band in low-volatility ranging markets (ADX<25)
# Enters short when 12h price touches upper band in low-volatility ranging markets (ADX<25)
# Uses BB width percentile filter to avoid choppy markets
# Exits when price touches middle band or ADX rises above 25 (trend developing)
# Designed for 12h timeframe with ~15-35 trades/year
name = "12h_BollingerRange_ADXFilter_V1"
timeframe = "12h"
leverage = 1.0