#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1d ADX trend filter.
# Long when price breaks above upper Donchian channel AND volume > 2.0x 50-period 1w average AND 1d ADX > 25 (trending market).
# Short when price breaks below lower Donchian channel AND volume > 2.0x 50-period 1w average AND 1d ADX > 25.
# Exit when price crosses the Donchian midpoint (upper+lower)/2.
# Uses discrete position size 0.25. Designed to capture breakouts in trending markets (both bull and bear).
# Target: 40-80 total trades over 4 years (10-20/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian(20) channels (from previous bar) ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Donchian channels: highest high and lowest low over 20 periods
    highest_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # === 1w Indicators: Volume Spike (volume > 2.0x 50-period average) ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=50, min_periods=50).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    # === 1d Indicators: ADX > 25 (trending market filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trending = adx_aligned > 25
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 100 periods needed for Donchian/ADX)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trending[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_trending = trending[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < donchian_mid[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > donchian_mid[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian AND volume spike AND trending market
            if price > donchian_upper[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower Donchian AND volume spike AND trending market
            elif price < donchian_lower[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wVolumeSpike_ADX_V1"
timeframe = "1d"
leverage = 1.0