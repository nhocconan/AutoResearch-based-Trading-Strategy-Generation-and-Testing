#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with daily trend filter and volume confirmation.
Uses daily EMA200 for trend direction and 12h Donchian channels for breakout entries.
In bull markets (price > daily EMA200): long on upper band breakout.
In bear markets (price < daily EMA200): short on lower band breakout.
Volume must be above 20-period average to confirm breakout.
Targets 50-150 total trades over 4 years with clear entry/exit rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)  # already shifted
    
    # === 12H DONCHIAN CHANNELS (LTF) ===
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        if np.isnan(daily_ema_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA
        bull_trend = close[i] > daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below lower band OR trend turns bearish
            if close[i] < lower[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper band OR trend turns bullish
            if close[i] > upper[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on daily trend
            if bull_trend:
                # In bull market: long on upper band breakout
                if close[i] > upper[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on lower band breakdown
                if close[i] < lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals