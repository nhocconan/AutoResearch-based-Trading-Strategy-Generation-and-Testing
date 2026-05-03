#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Uses 4h Donchian(20) for breakout signals, 1d EMA50 for trend direction, and volume spike for confirmation
# Session filter (08-20 UTC) reduces noise trades. Target: 15-37 trades/year (60-150 total over 4 years)
# Works in bull markets via breakouts in uptrend, bear markets via breakdowns in downtrend
# Prioritizes BTC/ETH performance with SOL as secondary

name = "1h_Donchian20_1dEMA50_VolumeConfirm_Session"
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
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Rolling max/min for Donchian channels
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_high = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 24-period EMA on 1h volume (approx 1 day)
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_24[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 24-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_24[i])
        
        # Donchian breakout with 1d EMA trend filter
        # Long: Price breaks above Donchian high + volume spike + price above 1d EMA50 (uptrend)
        # Short: Price breaks below Donchian low + volume spike + price below 1d EMA50 (downtrend)
        if position == 0:
            if (close[i] > donchian_high_aligned[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            elif (close[i] < donchian_low_aligned[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low (reversal) OR price below 1d EMA50 (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above Donchian high (reversal) OR price above 1d EMA50 (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals