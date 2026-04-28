#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX25 regime filter and volume confirmation
# Williams Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
# Long when Lips > Teeth > Jaw AND ADX > 25 (trending) AND volume > 1.5x 20-bar avg
# Short when Jaw > Teeth > Lips AND ADX > 25 AND volume > 1.5x 20-bar avg
# Exits when Alligator alignment breaks or volume drops
# Target: 19-50 trades/year via regime filter reducing whipsaw in ranging markets
# Works in both bull and bear markets by only trading when ADX confirms trending conditions

name = "4h_WilliamsAlligator_1dADX25_Regime_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed DM
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Prepend zeros for alignment (lost 27 bars: 1 for TR, 14 for TR smoothing, 12 for ADX smoothing)
    adx = np.concatenate([np.full(27, np.nan), adx])
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 4h data
    close_s = pd.Series(close)
    # Jaw: EMA(13) shifted by 8 bars
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full(len(jaw), np.nan)
    # Teeth: EMA(8) shifted by 5 bars
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full(len(teeth), np.nan)
    # Lips: EMA(5) shifted by 3 bars
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full(len(lips), np.nan)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        adx_val = adx_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Lips > Teeth > Jaw AND ADX > 25 (trending) AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and adx_val > 25 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Jaw > Teeth > Lips AND ADX > 25 (trending) AND volume confirmation
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and adx_val > 25 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) or ADX < 20 or no volume
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Alligator alignment breaks (Jaw <= Teeth or Teeth <= Lips) or ADX < 20 or no volume
            if jaw[i] <= teeth[i] or teeth[i] <= lips[i] or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals