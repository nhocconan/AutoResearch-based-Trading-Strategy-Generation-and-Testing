#!/usr/bin/env python3
"""
Experiment #099: 6h Camarilla Pivot + 1d Regime + Volume Spike (Novel)

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 1d ADX regime (trending vs ranging) and volume spikes (>1.5x average) 
capture high-probability moves. In ranging markets (ADX<25), fade extremes at R3/S3. 
In trending markets (ADX>25), breakout continuation at R4/S4. This adaptive approach 
works in both bull (trending breakouts) and bear (mean reversion in ranges, failed 
breakouts) markets. 6h timeframe targets 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_099_6h_camarilla_pivot_1d_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot and ADX regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === Calculate 1d Camarilla Pivot Levels (using prior day's OHLC) ===
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)  # Prior day close for reference
    
    if len(df_1d) >= 2:  # Need at least 2 days of data
        # Align 1d data to LTF index for shifting
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Prior day's OHLC using shift(1)
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        prior_day_range = prior_day_high - prior_day_low
        
        # Camarilla levels: Close + (Range * multiplier)
        camarilla_r3_vals = prior_day_close + (prior_day_range * 1.1000)
        camarilla_s3_vals = prior_day_close - (prior_day_range * 1.1000)
        camarilla_r4_vals = prior_day_close + (prior_day_range * 1.5000)
        camarilla_s4_vals = prior_day_close - (prior_day_range * 1.5000)
        
        # Create series aligned with 1d index
        camarilla_r3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r3_vals)
        camarilla_s3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s3_vals)
        camarilla_r4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r4_vals)
        camarilla_s4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s4_vals)
        camarilla_close_series = pd.Series(index=df_1d_indexed.index, data=prior_day_close)
        
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_series.values)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_series.values)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_series.values)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_series.values)
        camarilla_close_aligned = align_htf_to_ltf(prices, df_1d, camarilla_close_series.values)
    else:
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
        camarilla_close_aligned = np.full(n, np.nan)
    
    # === Calculate 1d ADX(14) for regime detection ===
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        n_local = len(close_arr)
        if n_local < period:
            return np.full(n_local, np.nan)
        
        # True Range
        tr = np.zeros(n_local)
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, n_local):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros(n_local)
        dm_minus = np.zeros(n_local)
        for i in range(1, n_local):
            up_move = high_arr[i] - high_arr[i-1]
            down_move = low_arr[i-1] - low_arr[i]
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0
        
        # Smoothed TR, DM+, DM-
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros(n_local)
        dx_denom = di_plus + di_minus
        mask = dx_denom != 0
        dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / dx_denom[mask]
        
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        adx_1d_raw = calculate_adx(high_1d, low_1d, close_1d, 14)
        
        # Align ADX to LTF
        adx_1d_series = pd.Series(index=df_1d.set_index('open_time').index, data=adx_1d_raw)
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_series.values)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF Camarilla, ADX, ATR, and volume
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: ADX > 25 = trending, ADX < 25 = ranging ---
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price relative to Camarilla levels ---
        price = close[i]
        near_r3 = abs(price - camarilla_r3_aligned[i]) < (0.001 * camarilla_r3_aligned[i])  # Within 0.1%
        near_s3 = abs(price - camarilla_s3_aligned[i]) < (0.001 * camarilla_s3_aligned[i])
        near_r4 = abs(price - camarilla_r4_aligned[i]) < (0.001 * camarilla_r4_aligned[i])
        near_s4 = abs(price - camarilla_s4_aligned[i]) < (0.001 * camarilla_s4_aligned[i])
        above_r4 = price > camarilla_r4_aligned[i]
        below_s4 = price < camarilla_s4_aligned[i]
        between_r3_s3 = (price > camarilla_s3_aligned[i]) and (price < camarilla_r3_aligned[i])
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if position_side > 0 and (price < camarilla_s3_aligned[i] or price > camarilla_r4_aligned[i]):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if position_side < 0 and (price > camarilla_r3_aligned[i] or price < camarilla_s4_aligned[i]):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Ranging market (ADX<25): Mean reversion at R3/S3
        long_ranging = is_ranging and volume_spike and near_s3 and (price > camarilla_close_aligned[i])
        short_ranging = is_ranging and volume_spike and near_r3 and (price < camarilla_close_aligned[i])
        
        # Trending market (ADX>25): Breakout continuation at R4/S4
        long_trending = is_trending and volume_spike and (near_r4 or above_r4) and (price > camarilla_close_aligned[i])
        short_trending = is_trending and volume_spike and (near_s4 or below_s4) and (price < camarilla_close_aligned[i])
        
        if long_ranging or long_trending:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_ranging or short_trending:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals