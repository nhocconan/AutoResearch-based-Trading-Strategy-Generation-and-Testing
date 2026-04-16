#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for HL Bands) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === HL Bands: 20-period high/low of 1d high/low ===
    # This creates adaptive support/resistance based on daily range
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    hl_high = high_series.rolling(window=20, min_periods=20).max().values  # 20-day high of daily highs
    hl_low = low_series.rolling(window=20, min_periods=20).min().values    # 20-day low of daily lows
    
    # === ADX for regime filter (14-period) ===
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nanmean(x[1:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1]/period) + x[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    # Align all HTF data to 6h timeframe
    hl_high_aligned = align_htf_to_ltf(prices, df_1d, hl_high)
    hl_low_aligned = align_htf_to_ltf(prices, df_1d, hl_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(hl_high_aligned[i]) or 
            np.isnan(hl_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        hl_high_val = hl_high_aligned[i]
        hl_low_val = hl_low_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_6h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below HL low OR reaches HL high (full range)
            if price < hl_low_val or price > hl_high_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above HL high OR reaches HL low
            if price > hl_high_val or price < hl_low_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Use HL band position for entry signals
            band_position = (price - hl_low_val) / (hl_high_val - hl_low_val) if hl_high_val > hl_low_val else 0.5
            
            if adx_val > 25:  # Trending regime: breakout at extremes
                # LONG: Break above upper band with volume
                if band_position > 0.95 and vol_ratio > 1.3:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Break below lower band with volume
                elif band_position < 0.05 and vol_ratio > 1.3:
                    signals[i] = -0.25
                    position = -1
                    continue
            else:  # Ranging regime: mean reversion at extremes
                # LONG: Near lower band with volume exhaustion
                if band_position < 0.15 and vol_ratio < 0.8:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Near upper band with volume exhaustion
                elif band_position > 0.85 and vol_ratio < 0.8:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_HL_Bands_ADX_Volume"
timeframe = "6h"
leverage = 1.0