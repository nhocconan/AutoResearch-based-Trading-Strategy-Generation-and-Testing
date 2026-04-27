#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot-based trend filter and volatility-adjusted breakout
# Uses 1-week high/low as trend filter (bullish if price > weekly mid-point, bearish if <)
# Combines with 6-hour ATR breakout for entry and volatility-based exit
# Designed to work in both bull (breakout continuation) and bear (mean reversion at extremes) markets
# Target: 15-30 trades/year to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly high/low for trend determination
    weekly_high = pd.Series(high_1w).expanding().max().values
    weekly_low = pd.Series(low_1w).expanding().min().values
    weekly_mid = (weekly_high + weekly_low) / 2
    
    # Align weekly trend to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Calculate 6h ATR(14) for volatility normalization
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-hour Donchian-like channels using ATR for dynamic breakout levels
    # Upper band: recent high + 0.5 * ATR
    # Lower band: recent low - 0.5 * ATR
    lookback = 10
    recent_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    recent_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    upper_band = recent_high + 0.5 * atr
    lower_band = recent_low - 0.5 * atr
    
    # Precompute session filter (08-20 UTC) - avoids low volatility Asian session
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(weekly_mid_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly midpoint
        price_above_weekly_mid = close[i] > weekly_mid_aligned[i]
        price_below_weekly_mid = close[i] < weekly_mid_aligned[i]
        
        # Breakout signals: price breaks ATR-adjusted bands
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Mean reversion signals: price returns to weekly midpoint area
        # Define reversion zone as ±0.25 * ATR around weekly midpoint
        reversion_zone_up = weekly_mid_aligned[i] + 0.25 * atr[i]
        reversion_zone_down = weekly_mid_aligned[i] - 0.25 * atr[i]
        mean_revert_up = close[i] < reversion_zone_up and position == -1  # covering short
        mean_revert_down = close[i] > reversion_zone_down and position == 1  # exiting long
        
        # Long conditions: 
        # 1. Bullish trend (price above weekly mid) + upward breakout
        # 2. Mean reversion from short when price approaches weekly mid from below
        long_breakout = price_above_weekly_mid and breakout_up
        long_reversion = price_below_weekly_mid and mean_revert_down
        
        # Short conditions:
        # 1. Bearish trend (price below weekly mid) + downward breakout  
        # 2. Mean reversion from long when price approaches weekly mid from above
        short_breakout = price_below_weekly_mid and breakout_down
        short_reversion = price_above_weekly_mid and mean_revert_up
        
        if (long_breakout or long_reversion) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_breakout or short_reversion) and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: 
        # - Trend reversal (price crosses weekly mid with conviction)
        # - Volatility collapse (ATR drops significantly suggesting range)
        elif position == 1 and (close[i] < weekly_mid_aligned[i] - 0.1 * atr[i] or atr[i] < atr[i-1] * 0.5):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > weekly_mid_aligned[i] + 0.1 * atr[i] or atr[i] < atr[i-1] * 0.5):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_ATRBreakout_MeanReversion"
timeframe = "6h"
leverage = 1.0