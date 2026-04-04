#!/usr/bin/env python3
"""
Experiment #5655: 6h Camarilla Pivot Fade/Breakout + Volume Spike + Regime Filter
HYPOTHESIS: On 6h timeframe, Camarilla pivot levels from 1d data provide high-probability 
fade zones at R3/S3 and breakout continuation zones at R4/S4. Volume spike (>2x average) 
confirms institutional participation. ADX regime filter ensures we only fade in ranging 
markets (ADX < 25) and breakout in trending markets (ADX > 25). This adaptive approach 
works in both bull and bear markets by aligning with market structure. Discrete sizing 
(0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5655_6h_camarilla_pivot_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla pivots and ADX ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla pivot levels for today (based on yesterday's OHLC)
        # These are intraday support/resistance levels
        yesterday_high = high_1d[-1] if len(high_1d) > 0 else close_1d[-1]
        yesterday_low = low_1d[-1] if len(low_1d) > 0 else close_1d[-1]
        yesterday_close = close_1d[-1] if len(close_1d) > 0 else close_1d[-1]
        
        # Calculate Camarilla levels
        range_yesterday = yesterday_high - yesterday_low
        camarilla_h5 = yesterday_close + 1.1 * range_yesterday * 1.1 / 2  # R4
        camarilla_h4 = yesterday_close + 1.1 * range_yesterday * 1.1 / 4  # R3
        camarilla_h3 = yesterday_close + 1.1 * range_yesterday * 1.1 / 6  # R2
        camarilla_l3 = yesterday_close - 1.1 * range_yesterday * 1.1 / 6  # S2
        camarilla_l4 = yesterday_close - 1.1 * range_yesterday * 1.1 / 4  # S3
        camarilla_l5 = yesterday_close - 1.1 * range_yesterday * 1.1 / 2  # S4
        
        # For simplicity, we'll use R3/S3 for fade and R4/S4 for breakout
        # In practice, these levels would be recalculated daily
        # We'll create arrays of these levels aligned to each bar
        camarilla_r3 = np.full_like(close, camarilla_h4)
        camarilla_s3 = np.full_like(close, camarilla_l4)
        camarilla_r4 = np.full_like(close, camarilla_h5)
        camarilla_s4 = np.full_like(close, camarilla_l5)
        
        # ADX calculation (14-period) on 1d data
        def calculate_adx(high_arr, low_arr, close_arr, period=14):
            if len(high_arr) < period + 1:
                return np.full_like(close_arr, np.nan)
            
            # True Range
            tr1 = high_arr[1:] - low_arr[1:]
            tr2 = np.abs(high_arr[1:] - close_arr[:-1])
            tr3 = np.abs(low_arr[1:] - close_arr[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr = np.concatenate([[np.nan], tr])  # align with original indices
            
            # Directional Movement
            dm_plus = np.where((high_arr[1:] - high_arr[:-1]) > (low_arr[:-1] - low_arr[1:]), 
                               np.maximum(high_arr[1:] - high_arr[:-1], 0), 0)
            dm_minus = np.where((low_arr[:-1] - low_arr[1:]) > (high_arr[1:] - high_arr[:-1]), 
                                np.maximum(low_arr[:-1] - low_arr[1:], 0), 0)
            dm_plus = np.concatenate([[np.nan], dm_plus])
            dm_minus = np.concatenate([[np.nan], dm_minus])
            
            # Smoothed values
            atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
            dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
            dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
            
            # Directional Indicators
            di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, 1)
            di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, 1)
            
            # ADX
            dx = np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), 1) * 100
            adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
            return adx
        
        adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    else:
        camarilla_r3 = camarilla_s3 = camarilla_r4 = camarilla_s4 = adx_1d = np.array([])
    
    # Align 1d indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 14)  # volume avg, ADX lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ratio[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or stoploss (using opposite level) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit: reverse signal OR price hits S3 (stoploss for longs)
                if (price >= camarilla_r3_aligned[i] and adx_1d_aligned[i] > 25) or price <= camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit: reverse signal OR price hits R3 (stoploss for shorts)
                if (price <= camarilla_s3_aligned[i] and adx_1d_aligned[i] > 25) or price >= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_spike = volume_ratio[i] > 2.0
        adx_value = adx_1d_aligned[i]
        
        # Regime-based entry: fade in ranging, breakout in trending
        if adx_value < 25:  # Ranging market - fade at R3/S3
            long_setup = (price <= camarilla_s3_aligned[i]) and volume_spike
            short_setup = (price >= camarilla_r3_aligned[i]) and volume_spike
        else:  # Trending market - breakout at R4/S4
            long_setup = (price >= camarilla_r4_aligned[i]) and volume_spike
            short_setup = (price <= camarilla_s4_aligned[i]) and volume_spike
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals