#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume spike and ADX trend filter.
# Long when price breaks above 20-period Donchian upper band AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Short when price breaks below 20-period Donchian lower band AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit when price reverts to 10-period Donchian midpoint OR ATR-based stoploss (2.5*ATR from entry).
# Uses discrete position size 0.30. Designed to capture strong trending moves with volume confirmation in both bull and bear markets.
# Avoids ranging markets via ADX filter and reduces false breakouts with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian channels ===
    donchian_len = 20
    donchian_mid_len = 10
    
    # Upper band: highest high over donchian_len periods
    highest_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    # Lower band: lowest low over donchian_len periods
    lowest_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    # Midpoint: average of upper and lower bands
    donchian_mid = (highest_high + lowest_low) / 2.0
    # Shorter midpoint for exit: average over donchian_mid_len periods
    highest_high_mid = pd.Series(high).rolling(window=donchian_mid_len, min_periods=donchian_mid_len).max().values
    lowest_low_mid = pd.Series(low).rolling(window=donchian_mid_len, min_periods=donchian_mid_len).min().values
    donchian_mid_exit = (highest_high_mid + lowest_low_mid) / 2.0
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1d Indicators: ADX > 25 (strong trending market filter) ===
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
    strong_trend = adx_aligned > 25
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 12h ATR for stoploss
    tr1_12h = pd.Series(high).diff()
    tr2_12h = pd.Series(low).diff().abs()
    tr3_12h = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_spike[i]) or
            np.isnan(strong_trend[i]) or np.isnan(atr_12h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverts to 10-period Donchian midpoint
            if price <= donchian_mid_exit[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverts to 10-period Donchian midpoint
            if price >= donchian_mid_exit[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 20-period Donchian upper band AND volume spike AND strong trending market
            if price > highest_high[i] and vol_spike and is_strong_trend:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below 20-period Donchian lower band AND volume spike AND strong trending market
            elif price < lowest_low[i] and vol_spike and is_strong_trend:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_1dADX_V1"
timeframe = "12h"
leverage = 1.0