#!/usr/bin/env python3
"""
1d_WeeklyTrixTrend_WithVolumeSpike
Hypothesis: TRIX momentum on weekly timeframe captures trend changes with low lag.
Combined with daily volume spike and price above/below weekly EMA34 for filtering.
Trades only in direction of weekly trend to avoid counter-trend whipsaws.
Targets 15-25 trades/year to minimize fee drag while capturing major moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for TRIX and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of log returns)
    close_1w = df_1w['close'].values
    log_returns = np.log(close_1w[1:] / close_1w[:-1])
    log_returns = np.concatenate([[0], log_returns])  # align length
    
    ema1 = pd.Series(log_returns).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 - pd.Series(ema3).shift(1))  # percentage change
    trix = trix.fillna(0).values
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily volume confirmation: >2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below EMA34
        weekly_bullish = close[i] > ema34_1w_aligned[i]
        weekly_bearish = close[i] < ema34_1w_aligned[i]
        
        # TRIX signal: zero cross with momentum
        trix_cross_up = trix_aligned[i-1] <= 0 and trix_aligned[i] > 0
        trix_cross_down = trix_aligned[i-1] >= 0 and trix_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: TRIX cross in direction of weekly trend with volume
        long_entry = vol_confirm and weekly_bullish and trix_cross_up
        short_entry = vol_confirm and weekly_bearish and trix_cross_down
        
        # Exit logic: opposite TRIX cross or trend change
        long_exit = trix_cross_down or (not weekly_bullish)
        short_exit = trix_cross_up or (not weekly_bearish)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyTrixTrend_WithVolumeSpike"
timeframe = "1d"
leverage = 1.0