#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA50 trend filter
# Long when price breaks above upper Donchian channel AND volume > 2.0x 20-period average AND 1w EMA50 rising
# Short when price breaks below lower Donchian channel AND volume > 2.0x 20-period average AND 1w EMA50 falling
# Exit when price crosses back to midpoint of Donchian channel OR 1w EMA50 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-35 trades/year per symbol.
# Donchian provides clear trend structure, volume spike confirms institutional interest,
# 1w EMA50 filters for primary trend to avoid counter-trend whipsaws in bear markets.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_1wEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate volume spike filter on 1d data: volume > 2.0x 20-period average
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_spike_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w data
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA50 > previous EMA50
    uptrend_1w = ema_50 > ema_50_prev
    downtrend_1w = ema_50 < ema_50_prev
    
    # Align 1w trend to 4h timeframe
    uptrend_1w_4h = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_4h = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Calculate Donchian(20) channels on primary 4h timeframe
    if len(high) >= 20:
        # Upper channel: highest high over last 20 periods
        upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower channel: lowest low over last 20 periods
        lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Midpoint for exit
        midpoint = (upper_channel + lower_channel) / 2.0
    else:
        upper_channel = np.full(n, np.nan)
        lower_channel = np.full(n, np.nan)
        midpoint = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(midpoint[i]) or 
            np.isnan(volume_spike_4h[i]) or 
            np.isnan(uptrend_1w_4h[i]) or 
            np.isnan(downtrend_1w_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND volume spike AND 1w uptrend
            if (close[i] > upper_channel[i] and 
                volume_spike_4h[i] > 0.5 and 
                uptrend_1w_4h[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND volume spike AND 1w downtrend
            elif (close[i] < lower_channel[i] and 
                  volume_spike_4h[i] > 0.5 and 
                  downtrend_1w_4h[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to midpoint OR 1w trend flips to downtrend
            if (close[i] < midpoint[i] or 
                downtrend_1w_4h[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to midpoint OR 1w trend flips to uptrend
            if (close[i] > midpoint[i] or 
                uptrend_1w_4h[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals