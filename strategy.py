#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ADX regime filter.
# Long when price breaks above 20-period Donchian high AND 1d volume > 2.0x 20-period MA AND 1d ADX > 25.
# Short when price breaks below 20-period Donchian low AND 1d volume > 2.0x 20-period MA AND 1d ADX > 25.
# Exit when price returns to mid-channel (Donchian median) OR ADX < 20 (regime change to ranging).
# Uses 4h timeframe to target 75-200 trades over 4 years (19-50/year) with tight entry conditions.
# Donchian channels provide clear structure, volume confirms breakout strength, ADX filters for trending markets only.
# Designed to work in both bull (strong breakouts) and bear (strong breakdowns) markets by trading with the trend.

name = "4h_Donchian20_1dVolumeSpike_ADXTrend"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    vol_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1d volume > 2.0x 20-period volume MA
        # Note: we use the aligned 1d volume MA, but need current 1d volume for spike
        # Since we don't have current 1d volume aligned, we approximate using price action
        # Alternative: use 4h volume spike as proxy (simpler and avoids lookahead)
        volume_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume[i] > (volume_ma_20_4h[i] * 2.0)
        
        # 1d ADX conditions
        adx_trending = adx_1d_aligned[i] > 25
        adx_ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND trending AND session
            if close[i] > donchian_high[i] and volume_spike and adx_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND trending AND session
            elif close[i] < donchian_low[i] and volume_spike and adx_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to mid-channel OR ADX becomes ranging
            if close[i] < donchian_mid[i] or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to mid-channel OR ADX becomes ranging
            if close[i] > donchian_mid[i] or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals