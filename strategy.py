#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d volume spike and 1d ADX trend filter.
# Enter long when price breaks above 4h Donchian upper channel with 1d volume spike and ADX > 25.
# Enter short when price breaks below 4h Donchian lower channel with 1d volume spike and ADX > 25.
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years.
# Uses 4h for signal direction, 1h only for entry timing. Session filter 08-20 UTC to reduce noise.

name = "1h_DonchianBreakout_4hTrend_1dVolumeSpike_ADX25_v1"
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
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    n_4h = len(high_4h)
    upper_channel = np.full(n_4h, np.nan)
    lower_channel = np.full(n_4h, np.nan)
    
    for i in range(n_4h):
        if i >= 19:  # min_periods=20
            upper_channel[i] = np.max(high_4h[i-19:i+1])
            lower_channel[i] = np.min(low_4h[i-19:i+1])
    
    # Forward fill to get most recent channel levels
    upper_channel = pd.Series(upper_channel).ffill().values
    lower_channel = pd.Series(lower_channel).ffill().values
    
    # Align 4h Donchian levels to 1h timeframe with 1-bar delay for confirmation
    upper_channel_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Get 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_1d = df_1d['volume'].values
    volume_series_1d = pd.Series(volume_1d)
    volume_ma_20_1d = volume_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    if n_1d < 14:
        return np.zeros(n)
    
    # True Range
    tr0 = high_1d - low_1d
    tr1 = np.abs(high_1d - np.roll(close_1d, 1))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_sum = np.zeros(n_1d)
    plus_dm_sum = np.zeros(n_1d)
    minus_dm_sum = np.zeros(n_1d)
    
    # Initial values (first 14 periods)
    tr_sum[13] = np.sum(tr[0:14])
    plus_dm_sum[13] = np.sum(plus_dm[0:14])
    minus_dm_sum[13] = np.sum(minus_dm[0:14])
    
    # Wilder's smoothing
    for i in range(14, n_1d):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / 14) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / 14) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / 14) + minus_dm[i]
    
    # Directional Indicators
    plus_di = np.zeros(n_1d)
    minus_di = np.zeros(n_1d)
    dx = np.zeros(n_1d)
    
    for i in range(14, n_1d):
        if tr_sum[i] != 0:
            plus_di[i] = 100 * (plus_dm_sum[i] / tr_sum[i])
            minus_di[i] = 100 * (minus_dm_sum[i] / tr_sum[i])
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX (smoothed DX)
    adx = np.zeros(n_1d)
    if n_1d >= 27:  # need 14 for DX + 14 for smoothing
        dx_sum = np.sum(dx[14:28])  # first 14 DX values
        adx[27] = dx_sum / 14
        for i in range(28, n_1d):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Forward fill ADX and volume spike
    adx = pd.Series(adx).ffill().values
    volume_spike_1d = pd.Series(volume_spike_1d).ffill().values
    
    # Align 1d indicators to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is datetime64[ms], .hour works on DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume spike and trend filter
        long_breakout = close[i] > upper_channel_aligned[i] and volume_spike_1d_aligned[i] and adx_aligned[i] > 25
        short_breakout = close[i] < lower_channel_aligned[i] and volume_spike_1d_aligned[i] and adx_aligned[i] > 25
        
        # Exit conditions: opposite channel break or loss of trend
        long_exit = close[i] < lower_channel_aligned[i] or adx_aligned[i] < 20
        short_exit = close[i] > upper_channel_aligned[i] or adx_aligned[i] < 20
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals