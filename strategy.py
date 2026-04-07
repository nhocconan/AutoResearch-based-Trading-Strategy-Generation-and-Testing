#!/usr/bin/env python3
"""
1h Donchian Breakout with Volume and 4h/1d Trend Filter
Long when price breaks above Donchian(20) with volume expansion and 4h/1d uptrend
Short when price breaks below Donchian(20) with volume expansion and 4h/1d downtrend
Exit when price returns to Donchian midpoint
Uses 4h/1d for direction, 1h for entry timing to minimize whipsaw.
Target: 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian_breakout_volume_4h1d_trend_v1"
timeframe = "1h"
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
    
    # === Donchian Channel (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === Volume confirmation (20-period avg) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 4h trend filter (EMA 34) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d trend filter (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint
            if close[i] <= donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint
            if close[i] >= donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume expansion
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Both 4h and 1d must agree on trend
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1] and ema_1d_aligned[i] > ema_1d_aligned[i-1]
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1] and ema_1d_aligned[i] < ema_1d_aligned[i-1]
            
            # Entry: Donchian breakout with volume and trend alignment
            if close[i] > donch_high[i] and trend_up:
                # Breakout above upper band with uptrend -> long
                position = 1
                signals[i] = 0.20
            elif close[i] < donch_low[i] and trend_down:
                # Breakdown below lower band with downtrend -> short
                position = -1
                signals[i] = -0.20
    
    return signals