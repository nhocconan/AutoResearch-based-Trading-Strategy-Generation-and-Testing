#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based volatility breakout with weekly pivot direction and volume confirmation.
# Uses ATR(14) breakout from mean (close) to capture expansion moves.
# Weekly pivot provides directional bias (long above weekly PP, short below).
# Volume filter ensures breakouts have institutional participation.
# Designed for low-frequency, high-quality trades in both bull and bear markets.
# Target: 15-30 trades/year to minimize fee drag.
name = "6h_ATRBreakout_WeeklyPivot_Volume"
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
    
    # Get weekly data for pivot points (directional bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly Pivot Point = (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pp_6h = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # ATR(14) for volatility measurement and breakout threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6-period mean close for breakout reference (1 day of 6h bars)
    mean_close = pd.Series(close).rolling(window=6, min_periods=6).mean().values
    
    # Volume filter: spike above 2.0x 24-period average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 6, 24)  # Wait for ATR, mean close, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr[i]) or np.isnan(mean_close[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pp_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above mean + ATR, above weekly pivot, volume confirmation
            if (close[i] > mean_close[i] + atr[i] and 
                close[i] > weekly_pp_6h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below mean - ATR, below weekly pivot, volume confirmation
            elif (close[i] < mean_close[i] - atr[i] and 
                  close[i] < weekly_pp_6h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below mean - ATR or below weekly pivot
            if close[i] < mean_close[i] - atr[i] or close[i] < weekly_pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above mean + ATR or above weekly pivot
            if close[i] > mean_close[i] + atr[i] or close[i] > weekly_pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals