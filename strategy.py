#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and 1w ADX regime filter.
# Long when price > Alligator Jaw (13-period SMMA shifted 8) AND volume > 2.0x 20-period 1d average AND 1w ADX > 20 (trending market).
# Short when price < Alligator Lips (8-period SMMA shifted 5) AND volume > 2.0x 20-period 1d average AND 1w ADX > 20.
# Exit when price crosses Alligator Teeth (8-period SMMA shifted 5) or ATR-based stoploss (2.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture trending moves with Alligator alignment.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Works in both bull and bear markets by requiring ADX>20 and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Williams Alligator ===
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price = (high + low) / 2
    smma_jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(smma_jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    smma_teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(smma_teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    smma_lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    lips = np.roll(smma_lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # === 1d Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # === 1w Indicators: ADX > 20 (trending market filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1w).diff()
    dm_minus = pd.Series(low_1w).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    trending_market = adx_aligned > 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for Alligator/ADX)
    warmup = 100
    
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
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trending_market[i]) or np.isnan(atr_12h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_trending = trending_market[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Alligator Teeth
            if price < teeth[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Alligator Teeth
            if price > teeth[i]:
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
            # LONG: Price > Alligator Jaw AND volume spike AND trending market
            if price > jaw[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price < Alligator Lips AND volume spike AND trending market
            elif price < lips[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_1wADX_V1"
timeframe = "12h"
leverage = 1.0