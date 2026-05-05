#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA50 trend filter
# Long when price breaks above Donchian upper AND 1d volume > 2.0x 20-period average AND 1w EMA50 > EMA50_prev (uptrend)
# Short when price breaks below Donchian lower AND 1d volume > 2.0x 20-period average AND 1w EMA50 < EMA50_prev (downtrend)
# Exit when price crosses back to Donchian midline OR 1w EMA50 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-40 trades/year per symbol.
# Donchian provides structure, volume spike confirms participation, 1w EMA50 filters primary trend to avoid counter-trend whipsaws.
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
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data (using previous 20 periods)
    if len(high) >= 20:
        # Upper channel: highest high of previous 20 periods
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        # Lower channel: lowest low of previous 20 periods
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
        # Midline: average of upper and lower
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: 1d volume > 2.0x 20-period average (spike filter)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_spike_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
        # Uptrend when current EMA50 > previous EMA50
        uptrend_1w = ema_50 > ema_50_prev
        downtrend_1w = ema_50 < ema_50_prev
    else:
        uptrend_1w = np.zeros(len(df_1w), dtype=bool)
        downtrend_1w = np.zeros(len(df_1w), dtype=bool)
        ema_50_prev = np.zeros(len(df_1w))
    
    # Align 1w trend to 4h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND 1w uptrend
            if (close[i] > donchian_high[i] and 
                volume_spike_aligned[i] > 0.5 and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND 1w downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian midline OR 1w trend flips to downtrend
            if (close[i] < donchian_mid[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian midline OR 1w trend flips to uptrend
            if (close[i] > donchian_mid[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals