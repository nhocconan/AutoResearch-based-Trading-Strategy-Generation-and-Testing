#!/usr/bin/env python3
"""
6h Elder Ray Power + Weekly Trend + Volume Confirmation
Hypothesis: Elder Ray (bull/bear power) captures institutional buying/selling pressure.
In trending markets (weekly trend), strong power readings with volume confirm continuation.
Works in bull (buy power > 0 + uptrend) and bear (sell power < 0 + downtrend).
Target: 80-180 total trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_weekly_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly and daily data for trend alignment (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    # Weekly EMA(50) for long-term trend
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Daily EMA(200) for intermediate trend
    close_daily = df_daily['close'].values
    ema_200_daily = pd.Series(close_daily).ewm(span=200, adjust=False).mean().values
    ema_200_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_200_daily)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull power: high minus EMA
    bear_power = low - ema_13   # Bear power: low minus EMA
    
    # 6h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require high volume
    
    # 6h ATR(14) for stoploss
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
    start = 50  # For EMA50 weekly and EMA200 daily
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_50_weekly_aligned[i]) or np.isnan(ema_200_daily_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend alignment: weekly EMA50 vs daily EMA200
        # Both must agree for trend confirmation
        weekly_uptrend = close[i] > ema_50_weekly_aligned[i]
        daily_uptrend = ema_200_daily_aligned[i] > ema_200_daily_aligned[i-1] if i > 0 else False
        weekly_downtrend = close[i] < ema_50_weekly_aligned[i]
        daily_downtrend = ema_200_daily_aligned[i] < ema_200_daily_aligned[i-1] if i > 0 else False
        
        uptrend = weekly_uptrend and daily_uptrend
        downtrend = weekly_downtrend and daily_downtrend
        
        # Check exits
        if position == 1:  # long position
            # Exit: bull power weakening OR stoploss
            if (bull_power[i] <= 0 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bear power weakening OR stoploss
            if (bear_power[i] >= 0 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray power + trend alignment + volume
            long_setup = (bull_power[i] > 0 and uptrend and vol_filter[i])
            short_setup = (bear_power[i] < 0 and downtrend and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals