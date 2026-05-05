#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d trend filter.
# Long when price breaks above 4h Donchian upper (20) AND 1h volume > 1.5x 20-period average AND 1d close > 1d EMA50 (uptrend)
# Short when price breaks below 4h Donchian lower (20) AND 1h volume > 1.5x 20-period average AND 1d close < 1d EMA50 (downtrend)
# Exit when price returns to 4h Donchian midpoint OR 1d trend flips.
# Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) to reduce noise.
# Discrete sizing 0.20 to limit fee drag. Target: 15-35 trades/year per symbol.

name = "1h_Donchian20_VolumeSpike_1dEMA50_Trend_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid per-bar datetime ops
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) using previous close to avoid look-ahead
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Use rolling window on 4h data, but shift by 1 to use only completed bars
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Shift to ensure we only use completed 4h bars (no look-ahead)
    donchian_high = np.concatenate([[np.nan], high_roll[:-1]])  # t-1 Donchian high
    donchian_low = np.concatenate([[np.nan], low_roll[:-1]])   # t-1 Donchian low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe
    dh_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    dm_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50
    downtrend_1d = close_1d < ema_50
    
    # Align 1d trend to 1h timeframe
    up_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    down_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: 1h volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(dh_aligned[i]) or 
            np.isnan(dl_aligned[i]) or 
            np.isnan(dm_aligned[i]) or 
            np.isnan(up_aligned[i]) or 
            np.isnan(down_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high AND volume spike AND 1d uptrend
            if (close[i] > dh_aligned[i] and 
                volume_filter[i] and 
                up_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low AND volume spike AND 1d downtrend
            elif (close[i] < dl_aligned[i] and 
                  volume_filter[i] and 
                  down_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h Donchian midpoint OR 1d trend flips to downtrend
            if (close[i] < dm_aligned[i] or 
                down_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Donchian midpoint OR 1d trend flips to uptrend
            if (close[i] > dm_aligned[i] or 
                up_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals