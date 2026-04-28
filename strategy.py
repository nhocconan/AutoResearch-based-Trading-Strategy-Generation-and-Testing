# 12h_WeeklyGapFill_Momentum
# Hypothesis: Price gaps from weekly open to Friday close often fill on Monday/Tuesday due to weekend liquidity rebalancing. 
# Uses weekly gap as mean-reversion signal with 1d trend filter and volume confirmation to avoid false fills in strong trends.
# Works in bull/bear: gaps fill in ranging markets, trend filter prevents counter-trend trades during strong moves.
# Target: 20-40 trades/year on 12h timeframe to minimize fee drag.
# Uses weekly open/close and 1d EMA50, aligned to 12h chart. No look-ahead bias.

#!/usr/bin/env python3
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
    
    # Get weekly data for gap calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly gap percentage (close - open) / open
    weekly_gap_pct = (weekly_close - weekly_open) / weekly_open
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily indicators to 12h timeframe
    gap_aligned = align_htf_to_ltf(prices, df_1w, weekly_gap_pct)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(gap_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Gap fill conditions: large weekly gap (>1.5%) with mean reversion
        gap_long = gap_aligned[i] < -0.015  # Weekly gap down >1.5% -> expect fill up
        gap_short = gap_aligned[i] > 0.015   # Weekly gap up >1.5% -> expect fill down
        
        long_entry = gap_long and uptrend and vol_filter
        short_entry = gap_short and downtrend and vol_filter
        
        # Exit conditions: gap filled or trend reverses
        long_exit = gap_aligned[i] > -0.005 or not uptrend  # Gap nearly filled or trend change
        short_exit = gap_aligned[i] < 0.005 or not downtrend
        
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

name = "12h_WeeklyGapFill_Momentum"
timeframe = "12h"
leverage = 1.0