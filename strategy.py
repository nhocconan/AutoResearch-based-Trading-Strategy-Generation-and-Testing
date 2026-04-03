#!/usr/bin/env python3
"""
Experiment #035: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with weekly pivot direction from daily timeframe capture 
institutional momentum with structural support/resistance. Volume confirmation filters false breakouts. 
ATR stoploss (2.5x) and minimum holding period reduce churn. Designed for both bull and bear markets by 
following the weekly pivot trend while using 6h for precise entry/exit. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_035_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === Calculate weekly pivot points from 1d data ===
    # Weekly pivot: (Prior week's High + Low + Close) / 3
    # We use prior completed week's OHLC to avoid look-ahead
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1).values  # Prior week high
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1).values    # Prior week low
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().shift(1).values  # Prior week close
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Support and resistance levels
    r1 = (2 * weekly_pivot) - weekly_low
    s1 = (2 * weekly_pivot) - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r2 + (weekly_high - weekly_low))  # R4 = R3 + (R2-R1) simplified
    s4_aligned = align_htf_to_ltf(prices, df_1d, s2 - (weekly_high - weekly_low))  # S4 = S3 - (S2-S1) simplified
    
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
    
    warmup = 50  # Warmup for indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Weekly Pivot Trend: Determine bias from weekly pivot ---
        # Bullish bias: price above weekly pivot and above R3
        bullish_bias = (price > weekly_pivot_aligned[i]) and (price > r3_aligned[i])
        # Bearish bias: price below weekly pivot and below S3
        bearish_bias = (price < weekly_pivot_aligned[i]) and (price < s3_aligned[i])
        
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
                # Exit on opposite Donchian breakout with volume (profit taking)
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
                # Exit on opposite Donchian breakout with volume (profit taking)
                if breakout_up and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade when breakout aligns with weekly pivot bias
        if bullish_bias:
            # Long: Donchian breakout up AND volume spike AND bullish bias
            if breakout_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif bearish_bias:
            # Short: Donchian breakout down AND volume spike AND bearish bias
            if breakout_down and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Neutral zone (between S3 and R3), do not trade to avoid chop
            signals[i] = 0.0
    
    return signals