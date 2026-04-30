#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper (20), 1d EMA50 uptrend, and volume > 1.8x 20-bar avg.
# Short when price breaks below 4h Donchian lower (20), 1d EMA50 downtrend, and volume > 1.8x 20-bar avg.
# Exit on opposite Donchian level touch or when trend filter fails.
# Uses 4h for signal direction (Donchian breakout + trend), 1h only for entry timing precision.
# Session filter (08-20 UTC) to reduce noise trades. Target position size 0.20.
# Designed to work in both bull and bear markets via trend filter and volatility-adjusted breaks.

name = "1h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 4h Donchian high, uptrend (close > 1d EMA50), volume spike
            if (curr_close > curr_donchian_high and
                curr_close > curr_ema_50_1d and
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low, downtrend (close < 1d EMA50), volume spike
            elif (curr_close < curr_donchian_low and
                  curr_close < curr_ema_50_1d and
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price touches/drops below Donchian low OR trend fails
            if (curr_close <= curr_donchian_low or
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions: price touches/rises above Donchian high OR trend fails
            if (curr_close >= curr_donchian_high or
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals