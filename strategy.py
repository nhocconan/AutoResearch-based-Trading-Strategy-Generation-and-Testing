#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + session filter
# Donchian(20) on 4h provides clear trend direction and structure
# Long when price breaks above 4h Donchian upper with volume confirmation during active session (08-20 UTC)
# Short when price breaks below 4h Donchian lower with volume confirmation during active session
# Uses discrete position sizing 0.20 to target ~15-30 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, session filter avoids low-liquidity periods

name = "1h_4h_donchian_breakout_v1"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high_4h, 20)
    donchian_lower_20 = rolling_min(low_4h, 20)
    
    # Calculate 4h average volume (20-period)
    vol_s_4h = pd.Series(volume_4h)
    avg_vol_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_20)
    avg_vol_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_vol_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(avg_vol_4h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 4h volume (scaled)
        # Scale 4h avg volume to 1h equivalent (approximate: 4h has ~4x bars of 1h)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if price falls below Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit short if price rises above Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout strategy: enter on Donchian breakout with volume confirmation
            if close[i] > donchian_upper_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.20
            elif close[i] < donchian_lower_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.20
    
    return signals