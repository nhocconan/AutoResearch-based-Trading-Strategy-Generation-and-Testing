#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high and 1d close > 1d EMA50.
Short when price breaks below 20-period Donchian low and 1d close < 1d EMA50.
Volume must be above 20-period average to confirm breakout strength.
Exit when price crosses back through Donchian midpoint or trend reverses.
Position size: 0.25. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
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
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # === DONCHIAN CHANNELS (LTF) ===
    donchian_len = 20
    donchian_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(daily_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA
        bull_trend = close[i] > daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR trend turns bearish
            if close[i] < donchian_mid[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR trend turns bullish
            if close[i] > donchian_mid[i] or bull_trend:
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
                # In bull market: long when price breaks above Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short when price breaks below Donchian low
                if close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals