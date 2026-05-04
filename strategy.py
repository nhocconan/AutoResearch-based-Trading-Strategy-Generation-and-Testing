#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day high AND 1w close > 1w EMA34 (uptrend) AND volume > 2.0x 20-day volume EMA
# Short when price breaks below 20-day low AND 1w close < 1w EMA34 (downtrend) AND volume > 2.0x 20-day volume EMA
# Uses 1d for precise entry timing, 1w for trend direction to avoid counter-trend trades.
# Discrete sizing (0.25) to minimize fee churn. Target: 10-25 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "1d_Donchian20_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper band = highest high of last 20 days
    # Lower band = lowest low of last 20 days
    high_roll_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 1d timeframe (no shift needed as we're already on 1d)
    # But we need to use previous day's levels to avoid look-ahead
    donchian_high = np.roll(high_roll_20, 1)  # Shift by 1 to use previous day's high
    donchian_low = np.roll(low_roll_20, 1)    # Shift by 1 to use previous day's low
    donchian_high[0] = np.nan  # First value is invalid
    donchian_low[0] = np.nan
    
    # Calculate 20-day volume EMA for volume spike filter
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ema_20 * 2.0)
    
    # Get 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Align daily indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 1w uptrend AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 1w downtrend AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 1w trend changes to downtrend
            if (close[i] < donchian_low_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 1w trend changes to uptrend
            if (close[i] > donchian_high_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals