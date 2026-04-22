#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day volume confirmation and 1-week trend filter.
# Uses 4h Donchian channels (20-period) for breakout signals, confirmed by 1-day volume spike
# and filtered by 1-week EMA(50) trend direction. This structure has shown strong test performance
# in the database for SOLUSDT and ETHUSDT. Designed for low trade frequency (<50/year) to avoid
# fee drag, with discrete position sizing (0.25) to manage drawdown in volatile markets like 2022.
# Works in both bull and bear markets by only taking breakouts in the direction of the higher-timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid low-liquidity hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1-day data for volume confirmation and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day volume spike filter (20-period)
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > 2.0 * vol_ma20_1d
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 1-week EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 4-hour Donchian channels (20-period)
    # We need 4h high/low, so load 4h data
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) \
           or np.isnan(vol_spike_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + 1w uptrend
            if (close[i] > donchian_high_aligned[i] and 
                vol_spike_1d_aligned[i] > 0.5 and  # confirmed by previous day's volume spike
                close[i] > ema_50_1w_aligned[i]):   # in 1-week uptrend
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + 1w downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_spike_1d_aligned[i] > 0.5 and  # confirmed by previous day's volume spike
                  close[i] < ema_50_1w_aligned[i]):  # in 1-week downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to the opposite Donchian level or trend fails
            if position == 1:
                if (close[i] < donchian_low_aligned[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > donchian_high_aligned[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_1wEMA50_Trend"
timeframe = "4h"
leverage = 1.0