#!/usr/bin/env python3
"""
1h_Camarilla_H3L3_Breakout_1dTrendFilter_v1
Hypothesis: On 1h timeframe, take breakout trades at Camarilla H3/L3 levels only when aligned with 1d trend (price above/below EMA50). Uses volume confirmation (vol_ratio > 1.5) and UTC 08-20 session filter. Target: 20-40 trades/year by requiring tight confluence of Camarilla breakout, 1d trend alignment, and volume spike. Discrete sizing 0.20 to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    # Since we're on 1h timeframe, we'll use the 1d data to get daily OHLC
    # But we need to align it properly to each 1h bar
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    camarilla_H3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_L3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Calculate 1h volume ratio (current vs 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(60, 24)
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 1d trend (bullish = price above EMA50)
        htf_1d_bullish = close > ema_50_1d_aligned[i]  # Using current 1h close vs aligned 1d EMA50
        htf_1d_bearish = close < ema_50_1d_aligned[i]
        
        # Volume confirmation: need significant spike (vol_ratio > 1.5)
        volume_confirmed = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long setup: price breaks above H3 + 1d bullish trend + volume confirmation
            long_setup = (close[i] > camarilla_H3_aligned[i]) and htf_1d_bullish and volume_confirmed
            
            # Short setup: price breaks below L3 + 1d bearish trend + volume confirmation
            short_setup = (close[i] < camarilla_L3_aligned[i]) and htf_1d_bearish and volume_confirmed
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price returns below H3 OR 1d trend turns bearish
            if (close[i] < camarilla_H3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price returns above L3 OR 1d trend turns bullish
            if (close[i] > camarilla_L3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_1dTrendFilter_v1"
timeframe = "1h"
leverage = 1.0