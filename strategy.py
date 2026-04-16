# 4h_Pivot_R1S1_Breakout_Volume_RangeFilter_Strict
# Hypothesis: Daily pivot levels (R1/S1) act as key support/resistance. 
# In trending regimes (ADX>25), breakouts of R1/S1 with volume confirmation capture momentum.
# In ranging regimes (ADX<20), mean reversion at R2/S2 with volume exhaustion provides counter-trend entries.
# Uses 4h for execution, 1d for pivots and regime filter. Tight entry criteria to limit trades (~20-50/year).
# Works in bull/bear by adapting to regime, avoiding whipsaws in low-ADX chop.
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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for pivots and ADX) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Daily Pivot Levels (Standard) ===
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)
    r2 = np.zeros_like(close_1d)
    s2 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        pivot[i] = (h + l + c) / 3.0
        r1[i] = 2 * pivot[i] - l
        s1[i] = 2 * pivot[i] - h
        r2[i] = pivot[i] + (h - l)
        s2[i] = pivot[i] - (h - l)
    
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
    
    # === 4h volume ratio for confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    # Align all HTF data to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1 (trend) OR reaches R2 (profit target in range)
            if price < s1_val or (adx_val < 20 and price > r2_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 (trend) OR reaches S2 (profit target in range)
            if price > r1_val or (adx_val < 20 and price < s2_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trending regime (ADX > 25): breakout continuation
            if adx_val > 25:
                # LONG: Break above R1 with volume
                if price > r1_val and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Break below S1 with volume
                elif price < s1_val and vol_ratio > 1.5:
                    signals[i] = -0.25
                    position = -1
                    continue
            # Ranging regime (ADX < 20): mean reversion at extremes
            elif adx_val < 20:
                # LONG: Reversion from S2 with volume exhaustion
                if price < s2_val and vol_ratio < 0.7:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Reversion from R2 with volume exhaustion
                elif price > r2_val and vol_ratio < 0.7:
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

name = "4h_Pivot_R1S1_Breakout_Volume_RangeFilter_Strict"
timeframe = "4h"
leverage = 1.0