#!/usr/bin/env python3
"""
1h Momentum + Volume + 1d Trend Filter
Hypothesis: Capture intraday momentum with volume confirmation filtered by daily trend (price above/below 200 EMA). Works in bull/bear by aligning with higher timeframe direction. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_volume_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stop loss
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
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA200 on daily close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 198) / 200
    
    # Daily trend: above EMA200 = bullish, below = bearish
    daily_trend = np.where(close_1d > ema_1d, 1, -1)
    
    # Align daily trend to 1h timeframe
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # 20-period EMA for momentum (1h)
    ema_fast = np.full(n, np.nan)
    if n >= 20:
        ema_fast[19] = np.mean(close[:20])
        for i in range(20, n):
            ema_fast[i] = (close[i] * 2 + ema_fast[i-1] * 18) / 20
    
    # 50-period EMA for trend (1h)
    ema_slow = np.full(n, np.nan)
    if n >= 50:
        ema_slow[49] = np.mean(close[:50])
        for i in range(50, n):
            ema_slow[i] = (close[i] * 2 + ema_slow[i-1] * 48) / 50
    
    # Momentum: fast EMA above slow EMA = bullish momentum
    momentum = np.where(ema_fast > ema_slow, 1, -1)
    
    # 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for EMAs
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(daily_trend_aligned[i]) or 
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: momentum turns bearish OR against daily trend
            # Stoploss: price drops 2*ATR below entry
            if (momentum[i] == -1 or
                daily_trend_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: momentum turns bullish OR against daily trend
            # Stoploss: price rises 2*ATR above entry
            if (momentum[i] == 1 or
                daily_trend_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                # Entry conditions: momentum aligned with daily trend + volume
                if momentum[i] == 1 and daily_trend_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif momentum[i] == -1 and daily_trend_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.20
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