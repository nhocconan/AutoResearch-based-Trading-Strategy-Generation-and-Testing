#!/usr/bin/env python3
"""
4h Pivots + Volume + Trend - Breakout/Mean-Reversion Strategy
Hypothesis: Uses 1d pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with volume confirmation and 1d trend filter.
Works in bull (breakouts with trend) and bear (breakdowns with trend). Targets 75-200 trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_pivots_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Get 1d data for trend and pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend bias
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish (1), below = bearish (-1)
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Calculate daily pivot levels from previous day's OHLC (avoid look-ahead)
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    r4_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    # Calculate pivots using previous day's data
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
            p = (ph + pl + pc) / 3.0
            pivot_1d[i] = p
            r1_1d[i] = 2*p - pl
            s1_1d[i] = 2*p - ph
            r2_1d[i] = p + (ph - pl)
            s2_1d[i] = p - (ph - pl)
            r3_1d[i] = ph + 2*(p - pl)
            s3_1d[i] = pl - 2*(ph - p)
            r4_1d[i] = 3*p - 2*pl
            s4_1d[i] = 3*ph - 2*pl
    
    # Align 1d data to 4h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 (mean reversion) OR against 1d trend OR stoploss
            if (close[i] < s3_aligned[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R3 (mean reversion) OR against 1d trend OR stoploss
            if (close[i] > r3_aligned[i] or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries - minimum 12 bars flat between trades
            if bars_since_entry >= 12:
                # Breakout entries: R4/S4 with trend
                bull_breakout = close[i] > r4_aligned[i]
                bear_breakout = close[i] < s4_aligned[i]
                
                # Mean reversion entries: near S3/R3 with counter-trend
                near_s3 = abs(close[i] - s3_aligned[i]) < (r1_aligned[i] - s1_aligned[i]) * 0.3
                near_r3 = abs(close[i] - r3_aligned[i]) < (r1_aligned[i] - s1_aligned[i]) * 0.3
                
                # Long: breakout with trend OR mean reversion at S3 against trend
                if (bull_breakout and trend_bias_aligned[i] == 1 and volume_filter) or \
                   (near_s3 and trend_bias_aligned[i] == -1 and volume_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at R3 against trend
                elif (bear_breakout and trend_bias_aligned[i] == -1 and volume_filter) or \
                     (near_r3 and trend_bias_aligned[i] == 1 and volume_filter):
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