#!/usr/bin/env python3
"""
1d Donchian(20) breakout with weekly trend filter and volume confirmation
Hypothesis: Donchian breakouts on daily charts capture multi-day momentum, filtered by weekly trend for bias and daily volume for conviction. Works in bull (buy breakouts above weekly trend) and bear (sell breakdowns below weekly trend). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_weekly_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get weekly data for trend filter (SMA50)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # SMA50 on weekly close
    sma_50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        for i in range(50, len(close_weekly)):
            sma_50_weekly[i] = np.mean(close_weekly[i-50:i])
    
    # Weekly trend: above SMA50 = bullish, below = bearish
    trend_weekly = np.where(close_weekly > sma_50_weekly, 1, -1)
    
    # Align weekly trend to daily timeframe
    trend_weekly_aligned = align_htf_to_ltf(prices, df_weekly, trend_weekly)
    
    # Get daily data for volume confirmation
    volume_daily = volume  # already daily
    
    # 20-period average volume on daily
    vol_ma_daily = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_daily[i] = np.mean(volume_daily[i-20:i])
    
    # Donchian channels (20-period) from daily data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for Donchian and weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_weekly_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_daily[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day average volume
        volume_filter = volume_daily[i] > vol_ma_daily[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_weekly_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_weekly_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                # Breakout entries: upper/lower with weekly trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish weekly trend + volume
                if bull_breakout and trend_weekly_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly trend + volume
                elif bear_breakout and trend_weekly_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals