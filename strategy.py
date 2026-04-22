#!/usr/bin/env python3
"""
Hypothesis: 1-hour strategy using 4h Donchian breakout + volume confirmation + 1d EMA200 trend filter.
Long when price breaks above 4h Donchian high (20-period) with volume spike and price > 1d EMA200.
Short when price breaks below 4h Donchian low with volume spike and price < 1d EMA200.
Exit when price crosses back through Donchian mid-point (average of high/low).
Uses 4h/1d for signal direction, 1h only for entry timing precision to avoid look-ahead.
Designed for low trade frequency (<30/year) with volume confirmation reducing false breaks.
Works in bull markets via breakouts and bear via short breakdowns; EMA200 filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 20-period Donchian high/low on 4h close
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align to 1h timeframe (already delayed for completed 4h bar)
    donch_high_1h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_1h = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_1h = align_htf_to_ltf(prices, df_4h, donch_mid)
    
    # Load 1d data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 200-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above 1d EMA200
            if (close[i] > donch_high_1h[i] and vol_spike and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below Donchian low with volume spike and below 1d EMA200
            elif (close[i] < donch_low_1h[i] and vol_spike and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price crosses Donchian mid-point
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below mid-point
                if close[i] < donch_mid_1h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above mid-point
                if close[i] > donch_mid_1h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_DonchianBreakout_Volume_EMA200"
timeframe = "1h"
leverage = 1.0