#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d ATR-based volatility filter and 1d Williams %R mean-reversion.
# Enters only during 08-20 UTC session. Uses tight conditions to limit trades (~20-40/year).
# Williams %R (14) < -80 for long, > -20 for short, with 1d ATR(14) > 1.5 * ATR(50) for volatility filter.
# Trend filter: price > 1d EMA50 for long, price < 1d EMA50 for short.
# Designed to work in both bull (trend + mean reversion) and bear (mean reversion in range) markets.
name = "4h_1d_WilliamsR_ATR_Volatility"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams %R, ATR, and EMA50 (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    
    # ATR (14) and ATR (50) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR is undefined
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volatility filter: ATR(14) > 1.5 * ATR(50)
    volatility_filter = atr_14_aligned > (atr_50_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), volatility filter, price > EMA50
            if (williams_r_aligned[i] < -80 and 
                volatility_filter[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), volatility filter, price < EMA50
            elif (williams_r_aligned[i] > -20 and 
                  volatility_filter[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Williams %R > -20 (overbought) or price < EMA50
            if williams_r_aligned[i] > -20 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Williams %R < -80 (oversold) or price > EMA50
            if williams_r_aligned[i] < -80 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals