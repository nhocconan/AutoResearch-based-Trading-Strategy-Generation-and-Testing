#!/usr/bin/env python3
"""
Experiment #231: 6h Camarilla Pivot + Volume Spike + Regime Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 1d ADX regime (ADX>25 = trend, ADX<20 = range) and volume spikes (>1.8x) 
capture high-probability moves. In ranging markets (ADX<20), fade at R3/S3 with target at 
daily pivot. In trending markets (ADX>25), breakout continuation at R4/S4 with trailing 
stop. This adapts to both bull (breakouts) and bear (failed breaks reverse) markets. 
6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_231_6h_camarilla_pivot_volume_regime_v1"
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
    
    # Calculate daily Camarilla pivot levels from prior day's OHLC
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), 
    #            S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    daily_pivot = np.full(n, np.nan)  # Pivot = (H+L+C)/3
    
    if len(df_1d) >= 2:
        df_1d_indexed = df_1d.set_index('open_time')
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        
        # Calculate prior day's range
        prior_day_range = prior_day_high - prior_day_low
        
        # Camarilla levels
        prior_day_r3 = prior_day_close + (prior_day_range * 1.1 / 4)
        prior_day_s3 = prior_day_close - (prior_day_range * 1.1 / 4)
        prior_day_r4 = prior_day_close + (prior_day_range * 1.1 / 2)
        prior_day_s4 = prior_day_close - (prior_day_range * 1.1 / 2)
        prior_day_pivot = (prior_day_high + prior_day_low + prior_day_close) / 3.0
        
        # Create series aligned with 1d index
        camarilla_r3_series = pd.Series(index=df_1d_indexed.index, data=prior_day_r3)
        camarilla_s3_series = pd.Series(index=df_1d_indexed.index, data=prior_day_s3)
        camarilla_r4_series = pd.Series(index=df_1d_indexed.index, data=prior_day_r4)
        camarilla_s4_series = pd.Series(index=df_1d_indexed.index, data=prior_day_s4)
        daily_pivot_series = pd.Series(index=df_1d_indexed.index, data=prior_day_pivot)
        
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_series.values)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_series.values)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_series.values)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_series.values)
        daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_series.values)
    else:
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
        daily_pivot_aligned = np.full(n, np.nan)
    
    # Calculate 1d ADX(14) for regime filter
    adx_14 = np.full(n, np.nan)
    if len(df_1d) >= 30:  # Need enough data for ADX
        df_1d_indexed = df_1d.set_index('open_time')
        h = df_1d_indexed['high'].values
        l = df_1d_indexed['low'].values
        c = df_1d_indexed['close'].values
        
        # True Range
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        
        # Directional Movement
        up_move = h - np.roll(h, 1)
        down_move = np.roll(l, 1) - l
        up_move[0] = 0
        down_move[0] = 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm14 / tr14
        minus_di = 100 * minus_dm14 / tr14
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_values = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align ADX to LTF
        adx_series = pd.Series(index=df_1d_indexed.index, data=adx_values)
        adx_14 = align_htf_to_ltf(prices, df_1d, adx_series.values)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Ensure enough data for HTF indicators, ATR, and volume
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(daily_pivot_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(adx_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: ADX > 25 = trending, ADX < 20 = ranging ---
        is_trending = adx_14[i] > 25
        is_ranging = adx_14[i] < 20
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Price Levels ---
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        pivot = daily_pivot_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                lowest_since_entry = min(lowest_since_entry, low[i])
            else:  # Short
                highest_since_entry = max(highest_since_entry, high[i])
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # ATR-based stoploss (2*ATR)
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
                # Take profit at 3*ATR or Camarilla level
                if high[i] >= entry_price + 3.0 * atr_14[i] or high[i] >= r4:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
                # Take profit at 3*ATR or Camarilla level
                if low[i] <= entry_price - 3.0 * atr_14[i] or low[i] <= s4:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
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
        # Regime-dependent entry conditions
        if is_ranging:
            # In ranging markets: fade at R3/S3, target at daily pivot
            long_condition = (close[i] <= s3 and 
                            close[i] > s4 and  # Avoid breakdown through S4
                            volume_spike and 
                            close[i] > low[i-1])  # Bullish rejection candle
            
            short_condition = (close[i] >= r3 and 
                             close[i] < r4 and  # Avoid breakout through R4
                             volume_spike and 
                             close[i] < high[i-1])  # Bearish rejection candle
        else:  # is_trending
            # In trending markets: breakout continuation at R4/S4
            long_condition = (close[i] >= r4 and 
                            volume_spike and 
                            close[i] > open[i])  # Bullish breakout candle
            
            short_condition = (close[i] <= s4 and 
                             volume_spike and 
                             close[i] < open[i])  # Bearish breakout candle
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals