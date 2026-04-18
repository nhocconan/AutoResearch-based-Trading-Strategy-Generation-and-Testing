#!/usr/bin/env python3
"""
1d_WeeklyTrend_Filter
Daily trend following with weekly trend filter and volume confirmation.
- Primary signal: Price above/below daily EMA50 (trend)
- Weekly filter: Only trade in direction of weekly EMA34 (avoid counter-trend)
- Entry trigger: Pullback to daily EMA20 with volume > 1.3x 20-day average
- Position size: 0.25 (25% of capital)
- Exit: Trend reversal (price crosses daily EMA50 opposite direction)
- Designed for 10-25 trades/year per symbol
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily EMA20 for entry timing
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need 50 for EMA50 + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        bull_weekly = close[i] > ema_34_1w_aligned[i]  # Price above weekly EMA34
        bear_weekly = close[i] < ema_34_1w_aligned[i]  # Price below weekly EMA34
        
        # Daily trend
        bull_daily = close[i] > ema_50_1d_aligned[i]
        bear_daily = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: pullback to EMA20 with volume
        near_ema20 = abs(close[i] - ema_20_1d_aligned[i]) / ema_20_1d_aligned[i] < 0.01  # within 1%
        volume_ok = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: weekly uptrend + daily uptrend + pullback to EMA20 + volume
            if bull_weekly and bull_daily and near_ema20 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + daily downtrend + pullback to EMA20 + volume
            elif bear_weekly and bear_daily and near_ema20 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns bearish OR price breaks below daily EMA50
            if not bull_weekly or bear_daily:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        
        elif position == -1:
            # Short exit: weekly trend turns bullish OR price breaks above daily EMA50
            if not bear_weekly or bull_daily:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals

name = "1d_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0