#!/usr/bin/env python3
"""
1h Range Breakout with 4h Trend Filter and Volume Confirmation
Long when price breaks above 1h range with 4h uptrend and high volume
Short when price breaks below 1h range with 4h downtrend and high volume
Exit when price returns to middle of 1h range
Designed for low-frequency, high-conviction trades to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_range_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-computed from DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4h Trend Filter: EMA(50) direction ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Determine 4h trend: slope of EMA
    ema_slope = np.zeros_like(ema_4h_aligned)
    ema_slope[1:] = ema_4h_aligned[1:] - ema_4h_aligned[:-1]
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    
    # === 1h Range: 20-period high/low ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    range_mid = (high_20 + low_20) / 2
    
    # === Volume filter: 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    vol_threshold = 1.5  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not in session
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to range midpoint
            if close[i] <= range_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to range midpoint
            if close[i] >= range_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need high volume
            if vol_ratio[i] < vol_threshold:
                signals[i] = 0.0
                continue
            
            # Long: break above range high with 4h uptrend
            if high[i] > high_20[i] and trend_up[i]:
                position = 1
                signals[i] = 0.20
            # Short: break below range low with 4h downtrend
            elif low[i] < low_20[i] and trend_down[i]:
                position = -1
                signals[i] = -0.20
    
    return signals