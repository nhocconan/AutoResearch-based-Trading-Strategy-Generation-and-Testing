#!/usr/bin/env python3
"""
4H Donchian Breakout + 12H Trend + Volume Confirmation + ATR Stop Loss
Breakout of 20-period Donchian channel with trend filter from 12h EMA.
Volume must exceed 20-period average for confirmation.
Exit when price crosses Donchian midline or trend reverses.
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
    
    # === 12H TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    twelve_h_close = df_12h['close'].values
    twelve_h_ema = pd.Series(twelve_h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    twelve_h_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_ema)
    
    # === 4H DONCHIAN CHANNEL (20-period) ===
    # Using pandas rolling on the price arrays
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(twelve_h_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        bull_trend = close[i] > twelve_h_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below midline OR trend turns bearish
            if close[i] < donchian_mid[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midline OR trend turns bullish
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
            
            # Entry logic: Donchian breakout with trend filter
            if bull_trend:
                # In bull trend: long on break above upper band
                if high[i] > donchian_high[i] and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear trend: short on break below lower band
                if low[i] < donchian_low[i] and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals