#!/usr/bin/env python3
# 4h_CBS_Breakout_Core - Close-Below-Stop Breakout
# Hypothesis: In strong trends (ADX>25), price breaking above/below the previous 4h bar's close with volume
# confirmation captures momentum. In weak trends (ADX<20), reversals at Bollinger Bands (20,2) with volume
# exhaustion provide mean reversion. Uses 4h for execution, 12h for ADX regime filter and Bollinger Bands.
# Target: 75-200 trades over 4 years (19-50/year) with disciplined entries.
# Works in both bull/bear by adapting to regime, avoiding whipsaws in low-ADX chop.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # === 12h data (HTF for regime filter and Bollinger Bands) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === Bollinger Bands (20,2) on 12h close ===
    bb_period = 20
    bb_std = 2
    sma_12h = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_12h = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_12h + bb_std * std_12h
    bb_lower = sma_12h - bb_std * std_12h
    
    # === ADX (14-period) on 12h ===
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
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
    
    # Align HTF data to 4h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Bollinger Bands and ADX
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        prev_close = close_4h[i//4] if i >= 4 else close[0]  # Previous 4h bar close (aligned)
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below previous 4h close (stop) OR reaches upper BB (target in range)
            if price < prev_close or (adx_val < 20 and price > bb_upper_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above previous 4h close (stop) OR reaches lower BB (target in range)
            if price > prev_close or (adx_val < 20 and price < bb_lower_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Regime-based entries
            if adx_val > 25:  # Trending regime: breakout of previous 4h close
                # LONG: Break above previous 4h close with volume
                if price > prev_close and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Break below previous 4h close with volume
                elif price < prev_close and vol_ratio > 1.5:
                    signals[i] = -0.25
                    position = -1
                    continue
            else:  # Ranging regime (ADX < 20): mean reversion at Bollinger Bands
                # LONG: Reversion from lower BB with volume exhaustion
                if price < bb_lower_val and vol_ratio < 0.7:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Reversion from upper BB with volume exhaustion
                elif price > bb_upper_val and vol_ratio < 0.7:
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

name = "4h_CBS_Breakout_Core"
timeframe = "4h"
leverage = 1.0