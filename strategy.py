#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Range Breakout with 1w Trend Filter and Volume Confirmation
# Uses daily timeframe to capture major price movements with minimal trade frequency.
# Breakouts occur when price moves beyond previous day's high/low with volume confirmation.
# Weekly trend filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Designed to work in both bull and bear markets by following weekly trend.
# Target: 20-50 trades per year to minimize fee drag while capturing significant moves.
name = "1d_DailyRangeBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Align to daily timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d_prev)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d_prev)
    
    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(ema_20_1d[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above previous day's high with volume spike and above weekly EMA20
            if (price > high_1d_aligned[i] and vol_spike[i] and price > ema_20_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day's low with volume spike and below weekly EMA20
            elif (price < low_1d_aligned[i] and vol_spike[i] and price < ema_20_1d[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below previous day's low (mean reversion)
            if price < low_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above previous day's high (mean reversion)
            if price > high_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals