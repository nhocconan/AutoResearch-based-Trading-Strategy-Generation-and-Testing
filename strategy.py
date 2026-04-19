#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, ADX > 25 (trending), and volume > 1.5x average.
# Short when price breaks below Donchian(20) low, ADX > 25, and volume > 1.5x average.
# Exit when price crosses Donchian midpoint or ADX < 20 (range).
# Designed for ~25-35 trades/year per symbol.
name = "4h_Donchian_ADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) calculation
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high, ADX > 25, volume > 1.5x average
            if price > donch_high_val and adx_val > 25 and vol_current > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, ADX > 25, volume > 1.5x average
            elif price < donch_low_val and adx_val > 25 and vol_current > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian midpoint or ADX < 20 (range)
            if price < donch_mid_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Donchian midpoint or ADX < 20 (range)
            if price > donch_mid_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals