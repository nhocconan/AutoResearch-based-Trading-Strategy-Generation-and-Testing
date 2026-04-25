#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyEMA50_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
Targets 15-25 trades/year by requiring: 1) price breaks 20-day Donchian channel, 2) aligned with weekly EMA50 trend,
3) volume > 1.8x 20-day average. Uses 1d timeframe to minimize fee drag while capturing significant moves.
Volume spike filter reduces false breakouts. Designed to work in both bull and bear markets by following the
weekly trend direction, avoiding counter-trend entries that fail in ranging/volatile conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-day Donchian channels (highest high/lowest low of past 20 days)
    # Using rolling window on daily data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) (20) + weekly EMA50 (50) + volume MA (20)
    start_idx = 50 + 20  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to weekly EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long breakout: price breaks above Donchian high with uptrend and volume confirmation
            long_breakout = (curr_high > donchian_high[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below Donchian low with downtrend and volume confirmation
            short_breakout = (curr_low < donchian_low[i]) and downtrend and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below Donchian low (mean reversion) or trend changes to downtrend
            if curr_low < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above Donchian high (mean reversion) or trend changes to uptrend
            if curr_high > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0