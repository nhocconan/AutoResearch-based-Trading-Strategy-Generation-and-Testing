#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w Supertrend filter and daily Donchian breakout.
# Long when price breaks above Donchian(20) upper band and price above weekly Supertrend.
# Short when price breaks below Donchian(20) lower band and price below weekly Supertrend.
# Uses weekly Supertrend to filter trend direction and avoid counter-trend trades.
# Target: 10-25 trades/year per symbol (30-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels (current day's data for breakout)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period) - use previous day's data to avoid look-ahead
    # For breakout at open of day, we use previous 20 days' high/low
    period = 20
    donchian_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    
    # Shift by 1 to use previous day's Donchian levels for today's breakout
    donchian_high_prev = np.roll(donchian_high, 1)
    donchian_low_prev = np.roll(donchian_low, 1)
    # Fill first value with current day's values (no look-ahead)
    donchian_high_prev[0] = high_1d[0]
    donchian_low_prev[0] = low_1d[0]
    
    # Align Donchian levels to 1d timeframe (already aligned, but using helper for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_prev)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_prev)
    
    # Load 1w data for Supertrend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Calculate ATR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate upper and lower bands
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    supertrend[:] = np.nan
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    if not np.isnan(atr[atr_period-1]):
        supertrend[atr_period-1] = upper_band[atr_period-1]
        direction[atr_period-1] = 1
    
    # Calculate Supertrend
    for i in range(atr_period, len(close_1w)):
        if np.isnan(supertrend[i-1]):
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close_1w[i] > supertrend[i-1]:
                supertrend[i] = upper_band[i]
                direction[i] = 1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = -1
    
    # Align Supertrend to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # 1d data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_donk = donchian_high_aligned[i]
        lower_donk = donchian_low_aligned[i]
        supertrend_val = supertrend_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band, above Supertrend, volume
            if price > upper_donk and price > supertrend_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, below Supertrend, volume
            elif price < lower_donk and price < supertrend_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower band or Supertrend turns bearish
            if price < lower_donk or price < supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper band or Supertrend turns bullish
            if price > upper_donk or price > supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Supertrend_Donchian_Breakout_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0