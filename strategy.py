#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and 1d Trend Filter
Long when price breaks above 12h Donchian high (20) with volume > 1.5x 20-period average and 1d EMA50 rising
Short when price breaks below 12h Donchian low (20) with volume > 1.5x 20-period average and 1d EMA50 falling
Exit when price crosses opposite Donchian band or volume drops below average
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Donchian channels (20-period) ===
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(n):
        if i < 19:
            donch_high[i] = np.nan
            donch_low[i] = np.nan
        else:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d trend filter (EMA50 slope) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        ema_1d = np.full(n, np.nan)
    else:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    ema1d_slope = np.zeros(n)
    ema1d_slope[:] = np.nan
    for i in range(1, n):
        if not np.isnan(ema_1d[i]) and not np.isnan(ema_1d[i-1]):
            ema1d_slope[i] = ema_1d[i] - ema_1d[i-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema1d_slope[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR volume drops below average
            if close[i] < donch_low[i] or vol_ratio[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR volume drops below average
            if close[i] > donch_high[i] or vol_ratio[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation and 1d trend filter
            if close[i] > donch_high[i] and ema1d_slope[i] > 0:
                # Price above Donchian high with rising 1d EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and ema1d_slope[i] < 0:
                # Price below Donchian low with falling 1d EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals