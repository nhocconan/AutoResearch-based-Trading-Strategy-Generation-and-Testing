#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 1d EMA34 trend filter
# Long when price breaks above Donchian upper(20) AND volume > 2.0x 20-period average AND 1d EMA34 > EMA34_prev (uptrend)
# Short when price breaks below Donchian lower(20) AND volume > 2.0x 20-period average AND 1d EMA34 < EMA34_prev (downtrend)
# Exit when price crosses back to Donchian midpoint OR 1d EMA34 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Donchian channels provide robust trend-following structure, volume spike confirms institutional participation,
# 1d EMA34 filters for primary trend direction to avoid counter-trend whipsaws in choppy markets.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_1dEMA34_Trend"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d data
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.concatenate([[np.nan], ema_34[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA34 > previous EMA34
    uptrend_1d = ema_34 > ema_34_prev
    downtrend_1d = ema_34 < ema_34_prev
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    if len(high) >= 20:
        # Rolling max/min for Donchian channels
        roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = roll_max
        donchian_lower = roll_min
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND 1d uptrend
            if (close[i] > donchian_upper[i] and 
                volume_filter[i] and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND 1d downtrend
            elif (close[i] < donchian_lower[i] and 
                  volume_filter[i] and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian midpoint OR 1d trend flips to downtrend
            if (close[i] < donchian_mid[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian midpoint OR 1d trend flips to uptrend
            if (close[i] > donchian_mid[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals