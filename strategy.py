#!/usr/bin/env python3
# 12h_donchian_volume_wave_v1
# Hypothesis: Uses Donchian channel breakouts on 12h timeframe with volume confirmation and trend filter from 1d EMA.
# Long when price breaks above 20-period Donchian high with volume > 1.5x average and price > 1d EMA50.
# Short when price breaks below 20-period Donchian low with volume > 1.5x average and price < 1d EMA50.
# Includes ATR-based stoploss and uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture sustained moves in both bull and bear markets by trading breakouts with institutional volume.
# Target: 15-30 trades/year (60-120 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_volume_wave_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for volatility filter and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 19 + tr[i]) / 20  # Wilder's smoothing
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            high_max[i] = np.max(high[i-19:i+1])
            low_min[i] = np.min(low[i-19:i+1])
    
    donchian_high = high_max
    donchian_low = low_min
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    
    # Calculate EMA with proper seeding
    k = 2 / (50 + 1)
    ema_50[49] = np.mean(close_1d[:50])  # Seed with SMA
    for i in range(50, len(close_1d)):
        ema_50[i] = close_1d[i] * k + ema_50[i-1] * (1 - k)
    
    # Align EMA50 to 12h timeframe (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation - 30 period average
    vol_ma_30 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 30:
            vol_sum -= volume[i-30]
        if i >= 29:
            vol_ma_30[i] = vol_sum / 30
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_30[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 30-period average
        vol_ok = volume[i] > vol_ma_30[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend fails
            if close[i] < donchian_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend fails
            if close[i] > donchian_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume confirmation and uptrend
            if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and vol_ok and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume confirmation and downtrend
            elif close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and vol_ok and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals