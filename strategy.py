#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA200 trend filter and volume confirmation.
- Trend: 12h EMA200 determines direction (bullish if price > EMA200, bearish if price < EMA200)
- Entry: In bull trend, long on breakout above 4h Donchian upper channel (20-period high)
         In bear trend, short on breakdown below 4h Donchian lower channel (20-period low)
- Volume: Must be above 20-period average to confirm breakout
- Exit: Reverse signal or volume drops below average
- Position sizing: 0.25 (25% of capital) to limit drawdown
- Expected trades: 20-50 per year (~80-200 over 4 years) due to strict trend+breakout+volume requirements
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # === 12h EMA200 TREND FILTER (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)  # already shifted
    
    # === 4h DONCHIAN CHANNEL (20-period) ===
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === VOLUME CONFIRMATION (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        if np.isnan(ema_12h_aligned[i]) or np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA200
        bull_trend = close[i] > ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: breakdown below lower Donchian band OR trend turns bearish OR volume weak
            if close[i] < low_4h[i] or not bull_trend or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: breakout above upper Donchian band OR trend turns bullish OR volume weak
            if close[i] > high_4h[i] or bull_trend or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on trend
            if bull_trend:
                # In bull trend: long on breakout above upper Donchian band
                if close[i] > high_4h[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear trend: short on breakdown below lower Donchian band
                if close[i] < low_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals