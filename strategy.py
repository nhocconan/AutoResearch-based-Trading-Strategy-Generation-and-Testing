#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(10) breakout with 1d volume confirmation and 6h ADX regime filter.
# Uses tighter Donchian period (10) to capture shorter-term breakouts while maintaining sufficient trade frequency.
# Long when price breaks above 10-period 4h high AND 1d volume > 1.2x 20-period average AND 6h ADX > 25 (strong trend).
# Short when price breaks below 10-period 4h low AND 1d volume > 1.2x 20-period average AND 6h ADX > 25.
# Exit when price crosses the 4h midpoint or ATR-based stop (1.5*ATR from entry).
# Discrete position size 0.25. Target: 80-150 total trades over 4 years (20-38/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (10-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper and lower bands (10-period)
    dc_upper_4h = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    dc_lower_4h = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    dc_mid_4h = (dc_upper_4h + dc_lower_4h) / 2
    
    # Align to 15m timeframe
    dc_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_4h)
    dc_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_4h)
    dc_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_mid_4h)
    
    # === 1d Indicators: Volume Spike (volume > 1.2x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    # === 6h Indicators: ADX > 25 (strong trend filter) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range
    tr1 = pd.Series(high_6h).diff()
    tr2 = pd.Series(low_6h).diff().abs()
    tr3 = pd.Series(close_6h).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_6h).diff()
    dm_minus = pd.Series(low_6h).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_6h, adx)
    strong_trend = adx_aligned > 25
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 4h ATR for stoploss
    tr1_4h = pd.Series(high_4h).diff()
    tr2_4h = pd.Series(low_4h).diff().abs()
    tr3_4h = pd.Series(close_4h).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
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
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > dc_mid_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND volume spike AND strong trend
            if price > dc_upper_4h_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND volume spike AND strong trend
            elif price < dc_lower_4h_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian10_1dVolumeSpike_6hADX_V1"
timeframe = "4h"
leverage = 1.0