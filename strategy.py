#!/usr/bin/env python3
"""
Experiment #067: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R3/S3 for fade, R4/S4 for continuation) 
and volume confirmation (>1.8x average) capture institutional flow with structural support/resistance. 
Weekly pivots from 1d HTF provide key levels that work in both bull/bear markets. Volume filter reduces false breakouts.
Target: 75-150 trades over 4 years (19-38/year) with signal size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_067_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week's 1d OHLC
    # Using rolling window of 5 days (1 week) to get prior week's high/low/close
    if len(df_1d) >= 5:
        # Shift by 1 to use prior week's data (avoid look-ahead)
        week_high = df_1d['high'].shift(1).rolling(window=5, min_periods=5).max().values
        week_low = df_1d['low'].shift(1).rolling(window=5, min_periods=5).min().values
        week_close = df_1d['close'].shift(1).rolling(window=5, min_periods=5).last().values
        
        # Calculate pivot point and support/resistance levels
        pivot = (week_high + week_low + week_close) / 3.0
        r1 = 2 * pivot - week_low
        s1 = 2 * pivot - week_high
        r2 = pivot + (week_high - week_low)
        s2 = pivot - (week_high - week_low)
        r3 = week_high + 2 * (pivot - week_low)
        s3 = week_low - 2 * (week_high - pivot)
        r4 = r3 + (r2 - r1)
        s4 = s3 - (s2 - s1)
        
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        # Not enough data for weekly calculation
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # === Weekly Pivot Zones ===
        # Fade zone: between S3 and R3 (mean reversion area)
        in_fade_zone = (price > s3_aligned[i]) and (price < r3_aligned[i])
        # Breakout zone: beyond R4 or S4 (strong momentum area)
        breakout_up_zone = price > r4_aligned[i]
        breakout_down_zone = price < s4_aligned[i]
        
        # === Volume Confirmation: Require volume spike (> 1.8x average) ===
        volume_spike = vol_ratio[i] > 1.8
        
        # === Donchian Breakout Conditions ===
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
        # === Exit Logic (ATR-based stoploss) ===
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
                # Take profit at R4 for longs
                if price > r4_aligned[i] and volume_spike:
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
                # Take profit at S4 for shorts
                if price < s4_aligned[i] and volume_spike:
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
        
        # === New Position Entry Logic (Only if Flat) ===
        # Long logic: Donchian breakout up in breakout zone OR fade zone with volume
        if breakout_up and volume_spike:
            if breakout_up_zone:
                # Strong continuation breakout above R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif in_fade_zone:
                # Fade trade from S3/R3 area back to pivot
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
        
        # Short logic: Donchian breakout down in breakout zone OR fade zone with volume
        elif breakout_down and volume_spike:
            if breakout_down_zone:
                # Strong continuation breakdown below S4
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif in_fade_zone:
                # Fade trade from R3/S3 area back to pivot
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals