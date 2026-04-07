#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with weekly trend filter and volume confirmation.
In bull market (weekly close > weekly EMA20): long on break above 1-day high.
In bear market (weekly close < weekly EMA20): short on break below 1-day low.
Volume must be above 20-period average to confirm.
Uses 1-day bars for Donchian channels and weekly trend.
Target: 10-30 total trades over 4 years (2-7/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)  # already shifted
    
    # === 1-DAY DONCHIAN CHANNELS ===
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        bull_trend = weekly_close[-1] > weekly_ema[-1] if len(weekly_close) > 0 else False
        # Use aligned weekly EMA for current bar
        bull_trend = weekly_ema_aligned[i] > 0  # placeholder, fix below
        # Actually compare close to ema
        bull_trend = close[i] > weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above donchian high OR trend turns bullish
            if close[i] > donchian_high[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on weekly trend and Donchian breakout
            if bull_trend:
                # In bull market: long on break above donchian high
                if high[i] > donchian_high[i] and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on break below donchian low
                if low[i] < donchian_low[i] and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals