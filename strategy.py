#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Confirmation
Long when price breaks above Donchian(20) high with weekly uptrend and volume surge
Short when price breaks below Donchian(20) low with weekly downtrend and volume surge
Exit when price crosses back through Donchian midpoint or trend reverses
Designed for low-frequency, high-conviction trades to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # === Donchian Channels (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === Weekly Trend (EMA 21) ===
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR weekly trend turns down
            if close[i] < donch_mid[i] or close[i] < weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR weekly trend turns up
            if close[i] > donch_mid[i] or close[i] > weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume surge (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with weekly trend alignment
            if close[i] > donch_high[i] and close[i] > weekly_ema_aligned[i]:
                # Break above upper band with weekly uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and close[i] < weekly_ema_aligned[i]:
                # Break below lower band with weekly downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals