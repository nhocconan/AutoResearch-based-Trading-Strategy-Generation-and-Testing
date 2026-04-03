#!/usr/bin/env python3
"""
Experiment #107: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (pivot, R1/S1, R2/S2) capture institutional order flow. Weekly pivots act as liquidity pools where price reacts. Volume confirmation (>1.5x average) filters weak breakouts. Works in bull/bear by trading both breakouts and mean reversion near pivots. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_107_6h_donchian20_1d_weekly_pivot_volume_v1"
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
    
    # Calculate weekly pivot from prior week's daily OHLC
    # We need to aggregate daily data into weekly bars
    # Since df_1d is already daily, we'll calculate weekly pivot from last 5 daily bars
    # But simpler: use prior week's high/low/close from daily data
    # We'll calculate rolling weekly pivot using last 5 daily bars (approx 1 week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot calculation
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly pivot levels (standard)
    r1 = weekly_pivot + weekly_range
    s1 = weekly_pivot - weekly_range
    r2 = weekly_pivot + 2 * weekly_range
    s2 = weekly_pivot - 2 * weekly_range
    r3 = weekly_pivot + 3 * weekly_range
    s3 = weekly_pivot - 3 * weekly_range
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
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
    
    warmup = 60  # Warmup for Donchian channels and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
        # --- Weekly Pivot Conditions ---
        # Near pivot levels for mean reversion
        near_pivot = abs(price - weekly_pivot_aligned[i]) / price < 0.003  # Within 0.3%
        near_r1 = abs(price - r1_aligned[i]) / price < 0.003
        near_s1 = abs(price - s1_aligned[i]) / price < 0.003
        near_r2 = abs(price - r2_aligned[i]) / price < 0.003
        near_s2 = abs(price - s2_aligned[i]) / price < 0.003
        
        # Breakout beyond pivot levels for continuation
        break_r2 = price > r2_aligned[i]
        break_s2 = price < s2_aligned[i]
        break_r3 = price > r3_aligned[i]
        break_s3 = price < s3_aligned[i]
        
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
                # Take profit if breaking S3 (strong support) or at 2R
                if break_s3 and volume_spike:
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
                # Take profit if breaking R3 (strong resistance) or at 2R
                if break_r3 and volume_spike:
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
        # Long conditions:
        # 1. Donchian breakout up AND volume spike AND near pivot/R1/S1 for mean reversion
        # 2. OR Donchian breakout up AND volume spike AND break above R2 for continuation
        # 3. OR break above R2 with volume (strong continuation signal)
        if ((breakout_up and volume_spike and (near_pivot or near_r1 or near_s1)) or
            (breakout_up and volume_spike and break_r2) or
            (break_r2 and volume_spike)):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short conditions:
        # 1. Donchian breakout down AND volume spike AND near pivot/R1/S1 for mean reversion
        # 2. OR Donchian breakout down AND volume spike AND break below S2 for continuation
        # 3. OR break below S2 with volume (strong continuation signal)
        elif ((breakout_down and volume_spike and (near_pivot or near_r1 or near_s1)) or
              (breakout_down and volume_spike and break_s2) or
              (break_s2 and volume_spike)):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals