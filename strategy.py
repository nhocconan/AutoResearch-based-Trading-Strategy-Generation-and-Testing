#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d ADX > 25 trend filter.
# Long when price breaks above upper Donchian channel AND 1d volume > 1.8x 20-period average AND 1d ADX > 25.
# Short when price breaks below lower Donchian channel AND 1d volume > 1.8x 20-period average AND 1d ADX > 25.
# Exit when price crosses the 12h midpoint (upper+lower)/2 or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong breakouts in trending markets.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_midpoint = (donchian_upper + donchian_lower) / 2
    
    # === 1d Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    # === 1d Indicators: ADX > 25 (trending market filter) ===
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
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_midpoint[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trending[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_trending = trending[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < donchian_midpoint[i]:
                exit_signal = True
            # Stoploss: 2*ATR below entry (using 1d ATR aligned)
            elif entry_price > 0 and price < entry_price - 2.0 * atr_1d[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > donchian_midpoint[i]:
                exit_signal = True
            # Stoploss: 2*ATR above entry (using 1d ATR aligned)
            elif entry_price > 0 and price > entry_price + 2.0 * atr_1d[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian AND volume spike AND trending market
            if price > donchian_upper[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower Donchian AND volume spike AND trending market
            elif price < donchian_lower[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            # Hold current position
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ADX_V1"
timeframe = "12h"
leverage = 1.0