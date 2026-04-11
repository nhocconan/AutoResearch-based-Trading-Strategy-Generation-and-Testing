#!/usr/bin/env python3
# 1h_4h_1d_camarilla_breakout_volume_v1
# Strategy: 1h Camarilla pivot breakout with 4h/1d trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong support/resistance. Breakouts above H3 or below L3 capture momentum.
# 4h EMA50 and 1d EMA50 filter ensures alignment with higher timeframe trend.
# Volume > 1.5x 20-period average confirms institutional participation.
# Designed for low trade frequency (~15-35/year) to minimize fee drift. Works in bull markets via long breakouts
# and bear markets via short breakdowns. Session filter (08-20 UTC) reduces noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h Camarilla pivot levels (based on previous day's OHLC)
    # We'll calculate daily pivots and align them to 1h
    # First get daily OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    daily_range = daily_high - daily_low
    H3 = daily_close + 1.0 * daily_range
    L3 = daily_close - 1.0 * daily_range
    
    # Align daily Camarilla levels to 1h timeframe
    H3_1h = align_htf_to_ltf(prices, df_1d, H3)
    L3_1h = align_htf_to_ltf(prices, df_1d, L3)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_1h[i]) or np.isnan(L3_1h[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Camarilla breakout signals
        breakout_up = high[i] > H3_1h[i-1]
        breakdown_down = low[i] < L3_1h[i-1]
        
        # Trend filters: price above EMA = bullish, below = bearish
        trend_4h_bullish = close[i] > ema_50_4h_aligned[i]
        trend_4h_bearish = close[i] < ema_50_4h_aligned[i]
        trend_1d_bullish = close[i] > ema_50_1d_aligned[i]
        trend_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Breakout above H3 AND both timeframes bullish AND volume confirmation
        if breakout_up and trend_4h_bullish and trend_1d_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Breakdown below L3 AND both timeframes bearish AND volume confirmation
        elif breakdown_down and trend_4h_bearish and trend_1d_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: Opposite Camarilla signal (breakdown for long, breakout for short)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals