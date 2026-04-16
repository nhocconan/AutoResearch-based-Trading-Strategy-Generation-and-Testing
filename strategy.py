#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) crossover with 1d volume confirmation and 1d ADX trend filter.
# Long when Alligator Lips (5-period SMA of median price, shifted 3) crosses above Teeth (8-period SMA, shifted 5) AND
# price is above Jaw (13-period SMA, shifted 8) AND 1d volume > 1.5x 20-period average AND 1d ADX > 20.
# Short when Lips crosses below Teeth AND price below Jaw AND same volume/ADX conditions.
# Exit when Lips crosses back over Teeth in opposite direction or price crosses Jaw.
# Uses discrete position size 0.25. Designed to catch trends with smoothed Alligator filters to reduce whipsaw.
# Williams Alligator is effective in both bull and bear markets by identifying trending vs ranging conditions.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Williams Alligator ===
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMA of median price, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA of median price, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA of median price, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1d Indicators: ADX > 20 (trending market filter) ===
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
    strong_trend = adx_aligned > 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max shift 8 + max period 13 = 21, plus ADX/ATR needs ~35)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Lips crosses below Teeth (trend weakening) OR price crosses below Jaw
            if lips_val < teeth_val or price < jaw_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Lips crosses above Teeth (trend weakening) OR price crosses above Jaw
            if lips_val > teeth_val or price > jaw_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Lips crosses above Teeth AND price above Jaw AND volume spike AND strong trend
            if lips_val > teeth_val and price > jaw_val and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Lips crosses below Teeth AND price below Jaw AND volume spike AND strong trend
            elif lips_val < teeth_val and price < jaw_val and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_1dADX_V1"
timeframe = "12h"
leverage = 1.0