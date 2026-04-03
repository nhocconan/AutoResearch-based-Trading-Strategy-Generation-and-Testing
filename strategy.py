#!/usr/bin/env python3
"""
Experiment #167: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) 
capture institutional order flow with reduced whipsaw. Weekly pivot levels derived from 1d OHLC provide significant 
support/resistance that price respects across multiple timeframes. Volume confirmation (1.5x 20-bar average) ensures 
participation. Discrete position sizing (0.25) and ATR trailing stop (2.0x) manage risk. Targets 12-25 trades/year 
on 6h timeframe to minimize fee drag. Works in bull/bear markets by using pivot levels as dynamic support/resistance 
and Donchian breakouts as momentum signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
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
    
    # Calculate weekly pivot levels from prior week's 1d OHLC
    # Weekly pivot: (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # R4 = Prior Week Close + 3*(Prior Week High - Prior Week Low)
    # R3 = Prior Week High + 2*(Prior Week Close - Prior Week Low)
    # S3 = Prior Week Low - 2*(Prior Week High - Prior Week Close)
    # S4 = Prior Week Close - 3*(Prior Week High - Prior Week Low)
    
    # Shift by 1 week (5 trading days) to use completed week's data
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(5).values
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(5).values
    week_close = df_1d['close'].shift(5).values
    
    pivot_point = (week_high + week_low + week_close) / 3.0
    r4 = week_close + 3.0 * (week_high - week_low)
    r3 = week_high + 2.0 * (week_close - week_low)
    s3 = week_low - 2.0 * (week_high - week_close)
    s4 = week_close - 3.0 * (week_high - week_low)
    
    # Align HTF pivot levels to 6h timeframe (auto shift(1) in align_htf_to_ltf)
    week_high_aligned = align_htf_to_ltf(prices, df_1d, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_1d, week_low)
    week_close_aligned = align_htf_to_ltf(prices, df_1d, week_close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Weekly Pivot Level Logic ---
        # Above R4: strong bullish continuation zone
        # Below S4: strong bearish continuation zone
        # Between R3 and S3: mean reversion zone (fade extremes)
        strong_bull_zone = close[i] > r4_aligned[i]
        strong_bear_zone = close[i] < s4_aligned[i]
        mean_revert_zone = (close[i] > s3_aligned[i]) & (close[i] < r3_aligned[i])
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: 
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price breaks below S4 OR touches lower Donchian in mean reversion zone
                    if close[i] < s4_aligned[i] or (mean_revert_zone and close[i] <= dc_lower_20[i]):
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price breaks above R4 OR touches upper Donchian in mean reversion zone
                    if close[i] > r4_aligned[i] or (mean_revert_zone and close[i] >= dc_upper_20[i]):
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian in strong bull zone OR mean reversion from S3 with volume
        if ((bullish_breakout and strong_bull_zone) or 
            (close[i] <= s3_aligned[i] and close[i] > dc_lower_20[i] and mean_revert_zone)) and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian in strong bear zone OR mean reversion from R3 with volume
        elif ((bearish_breakout and strong_bear_zone) or 
              (close[i] >= r3_aligned[i] and close[i] < dc_upper_20[i] and mean_revert_zone)) and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals