#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_Volume
# Hypothesis: On 12h timeframe, Camarilla pivot levels (R1/S1) from the prior day act as strong support/resistance.
# In trending regimes (ADX>25), breakouts of R1/S1 with volume confirmation capture momentum.
# In ranging regimes (ADX<25), mean reversion at R3/S3 with volume exhaustion provides counter-trend edges.
# Uses 12h for execution, 1d for Camarilla pivots and ADX regime filter.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.
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
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (HTF for Camarilla pivots and ADX) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Camarilla pivot levels (using previous day's OHLC) ===
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_r2 = np.zeros_like(close_1d)
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_pivot = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_s2 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        camarilla_pivot[i] = (h + l + c) / 3.0
        camarilla_r1[i] = c + ((h - l) * 1.1 / 12)
        camarilla_s1[i] = c - ((h - l) * 1.1 / 12)
        camarilla_r2[i] = c + ((h - l) * 1.1 / 6)
        camarilla_s2[i] = c - ((h - l) * 1.1 / 6)
        camarilla_r3[i] = c + ((h - l) * 1.1 / 4)
        camarilla_s3[i] = c - ((h - l) * 1.1 / 4)
        camarilla_r4[i] = c + ((h - l) * 1.1 / 2)
        camarilla_s4[i] = c - ((h - l) * 1.1 / 2)
    
    # === ADX for regime filter (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
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
    
    # === 12h volume ratio for confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    
    # Align all HTF data to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Camarilla and ADX calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1 (trend) OR reaches R3 (profit target in range)
            if price < s1 or (adx_val < 25 and price > r3):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 (trend) OR reaches S3 (profit target in range)
            if price > r1 or (adx_val < 25 and price < s3):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Regime-based entries
            if adx_val > 25:  # Trending regime: breakout continuation
                # LONG: Break above R1 with volume
                if price > r1 and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Break below S1 with volume
                elif price < s1 and vol_ratio > 1.5:
                    signals[i] = -0.25
                    position = -1
                    continue
            else:  # Ranging regime (ADX < 25): mean reversion at extremes
                # LONG: Reversion from S3 with volume exhaustion (volume < average)
                if price < s3 and vol_ratio < 0.7:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Reversion from R3 with volume exhaustion
                elif price > r3 and vol_ratio < 0.7:
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

name = "12h_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0