#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian Breakout with 1-day Trend Filter.
Long when price breaks above 20-period Donchian high in bull market (1d close > 1d EMA50).
Short when price breaks below 20-period Donchian low in bear market (1d close < 1d EMA50).
Volume must be above 20-period average to confirm breakouts.
Target: 15-40 trades per year (60-160 total over 4 years).
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
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === 12H DONCHIAN CHANNEL (LTF) ===
    dc_length = 20
    dc_high = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    dc_low = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish
            if close[i] < dc_low[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish
            if close[i] > dc_high[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if bull_trend:
                # In bull market: long on breakout above Donchian high
                if close[i] > dc_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on breakout below Donchian low
                if close[i] < dc_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals