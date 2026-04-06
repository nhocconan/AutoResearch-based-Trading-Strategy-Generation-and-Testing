#!/usr/bin/env python3
"""
1d Donchian(20) breakout with weekly trend filter and volume confirmation
Hypothesis: Daily breakouts capture medium-term momentum with low transaction costs.
Filter by weekly EMA200 for trend bias and daily volume > 1.5x 20-day average for conviction.
Works in bull (buy breakouts above weekly EMA200) and bear (sell breakdowns below weekly EMA200).
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
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
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA200 on weekly close
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 198) / 200
    
    # Weekly trend: above EMA200 = bullish, below = bearish
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Align weekly trend to daily timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Daily volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Donchian channels (20-period) from daily data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day average
        volume_filter = volume[i] > vol_ma_20[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against trend
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] < lower[i] or
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against trend
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] > upper[i] or
                trend_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries - minimum 3 days between trades
            if bars_since_exit >= 3:
                # Breakout entries: upper/lower with weekly trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish weekly trend + volume
                if bull_breakout and trend_1w_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: breakdown below lower with bearish weekly trend + volume
                elif bear_breakout and trend_1w_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals