#!/usr/bin/env python3
"""
Experiment #247: 6h Donchian(20) + 1d Weekly Pivot + Volume Confirmation

HYPOTHESIS: Donchian(20) breakout on 6h combined with 1d weekly pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) and volume confirmation captures strong directional moves. In trending markets, we trade breakouts in the direction of weekly pivot bias. Uses ATR-based stoploss and minimum 4-bar holding period to reduce churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_247_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d weekly pivot points (using prior week's OHLC)
    # We'll use rolling window of 5 days (1 week) to get prior week's H/L/C
    df_1d_index = pd.RangeIndex(len(df_1d))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior week's OHLC (shifted by 1 to avoid look-ahead)
    prior_week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    prior_week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    prior_week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Calculate weekly pivot points
    pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    r1 = 2 * pivot - prior_week_low
    s1 = 2 * pivot - prior_week_high
    r2 = pivot + (prior_week_high - prior_week_low)
    s2 = pivot - (prior_week_high - prior_week_low)
    r3 = prior_week_high + 2 * (pivot - prior_week_low)
    s3 = prior_week_low - 2 * (prior_week_high - pivot)
    r4 = prior_week_high + 3 * (prior_week_high - prior_week_low)
    s4 = prior_week_low - 3 * (prior_week_high - prior_week_low)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
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
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 205  # Warmup for 1d weekly pivot (5*24=120 6h bars per week + buffer)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Weekly Pivot Bias Determination ---
        # Bias: Above R4 = strong bullish, Below S4 = strong bearish
        # Between R3-S3 = neutral, R3-R4 = bullish bias, S3-S4 = bearish bias
        if price > r4_aligned[i]:
            pivot_bias = 2  # Strong bullish
        elif price > r3_aligned[i]:
            pivot_bias = 1  # Bullish bias
        elif price > s3_aligned[i]:
            pivot_bias = 0  # Neutral
        elif price > s4_aligned[i]:
            pivot_bias = -1  # Bearish bias
        else:
            pivot_bias = -2  # Strong bearish
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout with volume
                if breakout_down and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout with volume
                if breakout_up and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trade Donchian breakouts in direction of weekly pivot bias
        if breakout_up and volume_spike and pivot_bias >= 0:  # Long with bullish/neutral bias
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif breakout_down and volume_spike and pivot_bias <= 0:  # Short with bearish/neutral bias
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals