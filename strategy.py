#!/usr/bin/env python3
"""
1h_4h1d_KC_Donchian_Breakout_Volume_Trend
Hypothesis: Combine 4h Donchian breakout with 1d Keltner channel trend filter and volume confirmation on 1h timeframe.
The 4h Donchian provides institutional breakout signals, 1d Keltner channel filters for trend direction,
and volume confirmation ensures breakout validity. Works in bull/bear by following strong momentum.
Target: 15-37 trades/year (60-150 total over 4 years) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 1h timeframe
    donch_high_1h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_1h = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 1d data for Keltner channel trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Keltner channel (20-period)
    atr_1d = pd.Series(
        np.maximum(
            df_1d['high'] - df_1d['low'],
            np.maximum(
                abs(df_1d['high'] - df_1d['close'].shift(1)),
                abs(df_1d['low'] - df_1d['close'].shift(1))
            )
        )
    ).rolling(window=20, min_periods=20).mean().values
    
    keltner_mid = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = keltner_mid + (2 * atr_1d)
    keltner_lower = keltner_mid - (2 * atr_1d)
    
    # Align Keltner to 1h timeframe
    keltner_upper_1h = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_1h = align_htf_to_ltf(prices, df_1d, keltner_lower)
    keltner_mid_1h = align_htf_to_ltf(prices, df_1d, keltner_mid)
    
    # Volume filter: >1.8x 24-period average (more selective)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or
            np.isnan(keltner_upper_1h[i]) or np.isnan(keltner_lower_1h[i]) or
            np.isnan(keltner_mid_1h[i]) or np.isnan(volume_filter[i]) or
            np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        dh = donch_high_1h[i]
        dl = donch_low_1h[i]
        ku = keltner_upper_1h[i]
        kl = keltner_lower_1h[i]
        km = keltner_mid_1h[i]
        vol_ok = volume_filter[i]
        in_session = session_filter[i]
        
        if position == 0:
            # Long: break above 4h Donchian high with volume, above 1d Keltner mid (uptrend)
            if price > dh and vol_ok and price > km and in_session:
                signals[i] = 0.20
                position = 1
            # Short: break below 4h Donchian low with volume, below 1d Keltner mid (downtrend)
            elif price < dl and vol_ok and price < km and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 1d Keltner lower or Donchian low
            if price < kl or price < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 1d Keltner upper or Donchian high
            if price > ku or price > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_KC_Donchian_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0