#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d Trend Filter + Volume Confirmation
Long when price breaks above Donchian(20) high with 1d uptrend and volume spike
Short when price breaks below Donchian(20) low with 1d downtrend and volume spike
Exit when price crosses Donchian midpoint or trend reverses
Designed for 4h timeframe with controlled trade frequency (~25-40 trades/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # === 1d Trend Filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Donchian Channels (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === Volume Confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price crosses midpoint OR trend reverses
            if close[i] < donch_mid[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price crosses midpoint OR trend reverses
            if close[i] > donch_mid[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume filter: require significant volume spike
            if vol_ratio[i] < 1.8:
                signals[i] = 0.0
                continue
            
            # Long entry: break above Donchian high with 1d uptrend
            if close[i] > donch_high[i] and close[i] > ema_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: break below Donchian low with 1d downtrend
            elif close[i] < donch_low[i] and close[i] < ema_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals