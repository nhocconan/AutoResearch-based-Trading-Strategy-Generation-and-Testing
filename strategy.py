#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily Trend and Volume Confirmation
Hypothesis: On 12h timeframe, Donchian(20) breakouts in the direction of daily EMA(50) trend,
confirmed by volume spikes, capture significant moves in both bull and bear markets.
Daily trend filter avoids counter-trend trades. Volume ensures institutional participation.
ATR-based stoploss limits drawdown. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend and Donchian channels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Calculate rolling max/min for 20 periods
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1d, high_roll)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_roll)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)  # Require high volume
    
    # 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: break below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: break above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend alignment + volume
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            uptrend = ema_50_1d_aligned[i] > close_1d[i] if i < len(close_1d) else ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]  # Simplified trend check
            downtrend = ema_50_1d_aligned[i] < close_1d[i] if i < len(close_1d) else ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
            
            # Use 1d close for trend comparison (more reliable)
            if i < len(close_1d):
                uptrend = ema_50_1d_aligned[i] > close_1d[i]
                downtrend = ema_50_1d_aligned[i] < close_1d[i]
            else:
                # Fallback: use slope of EMA
                if i > 0:
                    uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
                    downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
                else:
                    uptrend = downtrend = False
            
            if long_breakout and uptrend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals