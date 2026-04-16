#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1d ADX trend filter.
# Long when price breaks above 20-period 4h high AND volume > 1.8x 20-period 1d average AND 1d ADX > 25.
# Short when price breaks below 20-period 4h low AND volume > 1.8x 20-period 1d average AND 1d ADX > 25.
# Exit when price crosses the 4h midpoint or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture major breakouts in strong trending markets.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper and lower bands (20-period)
    dc_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    dc_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    dc_mid_4h = (dc_upper_4h + dc_lower_4h) / 2
    
    # Align 4h to 4h timeframe (primary)
    dc_upper_4h_aligned = dc_upper_4h
    dc_lower_4h_aligned = dc_lower_4h
    dc_mid_4h_aligned = dc_mid_4h
    
    # === 1d Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
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
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 4h ATR for stoploss
    tr1_4h = pd.Series(high_4h).diff()
    tr2_4h = pd.Series(low_4h).diff().abs()
    tr3_4h = pd.Series(close_4h).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = atr_4h_raw  # Already aligned as primary timeframe
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(dc_upper_4h_aligned[i]) or np.isnan(dc_lower_4h_aligned[i]) or np.isnan(dc_mid_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or np.isnan(atr_4h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < dc_mid_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > dc_mid_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND volume spike AND strong trending market
            if price > dc_upper_4h_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND volume spike AND strong trending market
            elif price < dc_lower_4h_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_1dADX_V1"
timeframe = "4h"
leverage = 1.0