#!/usr/bin/env python3
"""
1h Overnight Gap Fade with 1d Trend Filter
Hypothesis: During low-volatility overnight hours (00-08 UTC), price gaps often fade back toward the prior day's VWAP.
In both bull and bear markets, mean reversion occurs after excessive overnight moves.
Uses 1d trend filter to fade gaps only in the direction of the higher timeframe trend.
Designed for low trade frequency (15-30/year) to minimize fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_overnight_gap_fade_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter and VWAP (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d VWAP (volume-weighted average price)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 1h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-calculate hour in UTC (already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # 1h ATR(14) for dynamic thresholds
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For 1d EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        
        # Overnight session: 00-08 UTC (low volatility, mean reversion prone)
        is_overnight = (0 <= hour < 8)
        
        # Day session: 08-20 UTC (active trading, follow trend)
        is_day = (8 <= hour < 20)
        
        # Check exits: mean reversion to VWAP or stoploss
        if position == 1:  # long position
            # Exit: price reverts to VWAP or stoploss hit
            if (close[i] >= vwap_1d_aligned[i] or 
                close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price reverts to VWAP or stoploss hit
            if (close[i] <= vwap_1d_aligned[i] or 
                close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: gap fade during overnight, trend filter during day
            overnight_entry = False
            day_entry = False
            
            if is_overnight:
                # Overnight: fade gaps > 1.5*ATR from VWAP
                gap_from_vwap = close[i] - vwap_1d_aligned[i]
                if gap_from_vwap < -1.5 * atr[i]:  # Gap down, go long
                    overnight_entry = True
                elif gap_from_vwap > 1.5 * atr[i]:  # Gap up, go short
                    overnight_entry = True
            
            if is_day:
                # Day: only trade in direction of 1d trend
                if close[i] > ema50_1d_aligned[i]:  # Uptrend
                    # Look for pullbacks to VWAP for longs
                    if close[i] < vwap_1d_aligned[i] and close[i] > vwap_1d_aligned[i] - 1.0 * atr[i]:
                        day_entry = True
                else:  # Downtrend
                    # Look for bounces to VWAP for shorts
                    if close[i] > vwap_1d_aligned[i] and close[i] < vwap_1d_aligned[i] + 1.0 * atr[i]:
                        day_entry = True
            
            if overnight_entry:
                if close[i] < vwap_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                else:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
            elif day_entry:
                if close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                else:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals