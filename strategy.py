#!/usr/bin/env python3
"""
Experiment #051: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Combining 6h Donchian breakouts with 1d weekly pivot levels (R3/S3 for fading, R4/S4 for breakout continuation) and volume confirmation captures institutional order flow at key weekly levels. Weekly pivots act as smart money reference points where price often accelerates after testing or breaks through. Volume spike confirms participation. Designed to work in both bull/bear markets by adapting to weekly structure. Targeting 75-200 trades over 4 years for statistical validity and fee efficiency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_051_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === Calculate weekly pivot points from prior week's OHLC ===
    # Need weekly OHLC - resample 1d to weekly using actual Binance weekly logic via shifting
    # Since we don't have weekly data directly, we'll approximate using prior 5 trading days
    # Weekly high = max(high of prior 5 days), weekly low = min(low of prior 5 days), weekly close = close of prior 5th day
    # We'll use rolling window of 5 days on 1d data
    if len(df_1d) >= 5:
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values  # last close in window
        weekly_open = pd.Series(df_1d['open'].values).rolling(window=5, min_periods=5).apply(lambda x: x[0]).values  # first open in window
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        # Camarilla-like weekly levels: R3, S3, R4, S4
        r3 = weekly_pivot + weekly_range * 1.1
        s3 = weekly_pivot - weekly_range * 1.1
        r4 = weekly_pivot + weekly_range * 1.3
        s4 = weekly_pivot - weekly_range * 1.3
    else:
        # Not enough data for weekly calc
        weekly_pivot = np.full(len(df_1d), np.nan)
        r3 = np.full(len(df_1d), np.nan)
        s3 = np.full(len(df_1d), np.nan)
        r4 = np.full(len(df_1d), np.nan)
        s4 = np.full(len(df_1d), np.nan)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
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
    
    warmup = 60  # Warmup for Donchian and weekly stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Weekly Pivot Context ---
        # Determine if we're in weekly accumulation/distribution zone
        near_s3 = abs(price - s3_aligned[i]) / s3_aligned[i] < 0.005  # Within 0.5% of S3
        near_r3 = abs(price - r3_aligned[i]) / r3_aligned[i] < 0.005  # Within 0.5% of R3
        breakout_r4 = price > r4_aligned[i]  # Break above R4
        breakdown_s4 = price < s4_aligned[i]  # Break below S4
        
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
                # Take profit at weekly R3/S3 test
                if position_side > 0 and near_r3 and volume_spike:
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
                # Take profit at weekly S3/R3 test
                if position_side < 0 and near_s3 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn (6h bars = 1 day)
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Fade at weekly S3/R3 with volume spike and Donchian breakout confirmation
        if near_s3 and breakout_up and volume_spike:
            # Long: price at S3 weekly support, breaking Donchian up with volume
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif near_r3 and breakout_down and volume_spike:
            # Short: price at R3 weekly resistance, breaking Donchian down with volume
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        # Breakout continuation at weekly R4/S4 with volume
        elif breakout_r4 and volume_spike:
            # Long: break above weekly R4 with volume
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif breakdown_s4 and volume_spike:
            # Short: break below weekly S4 with volume
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals