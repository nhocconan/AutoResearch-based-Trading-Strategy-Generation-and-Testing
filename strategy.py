#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian(20) breakout with 1-week trend filter and volume confirmation.
In bull market (1w close > 1w EMA50): long on break above Donchian(20) high.
In bear market (1w close < 1w EMA50): short on break below Donchian(20) low.
Volume must be above 20-period average to confirm breakout.
Targets 50-150 total trades over 4 years (12-37/year) with 0.25 position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1w_trend_volume_v1"
timeframe = "6h"
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
    
    # === 1W TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)  # already shifted
    
    # === DONCHIAN CHANNEL (LTF) ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(one_w_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        bull_trend = close[i] > one_w_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
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
            
            # Entry logic based on 1w trend and Donchian breakout
            if bull_trend:
                # In bull market: long on break above Donchian high
                if high[i] > donchian_high[i] and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on break below Donchian low
                if low[i] < donchian_low[i] and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals