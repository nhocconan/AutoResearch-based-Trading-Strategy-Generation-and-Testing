#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volatility filter and weekly trend filter
# - Uses 6h Donchian breakout (20-period) for entry signals
# - Filters with 1d ATR ratio (ATR(7)/ATR(30) < 0.8) to avoid high volatility chop
# - Uses 1w EMA(21) for trend filter: long only when price > EMA21, short only when price < EMA21
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture trending moves, volatility filter reduces false signals in chop
# - Weekly trend filter ensures we trade with the higher timeframe trend

name = "6h_1d_1w_donchian_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(7) and ATR(30) for volatility filter
    atr_7 = pd.Series(tr_1d).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    # Volatility filter: ATR(7)/ATR(30) < 0.8 (low volatility environment)
    vol_filter = (atr_7 / atr_30) < 0.8
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    # 1w EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d and 1w indicators to 6h
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 6h price data for Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vol_filter_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            close[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low
            if low[i] <= lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high
            if high[i] >= highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with filters
            # Long: price breaks above Donchian high + low volatility + price > weekly EMA
            if (high[i] >= highest_20[i] and
                vol_filter_aligned[i] and
                close[i] > ema_21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low + low volatility + price < weekly EMA
            elif (low[i] <= lowest_20[i] and
                  vol_filter_aligned[i] and
                  close[i] < ema_21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals