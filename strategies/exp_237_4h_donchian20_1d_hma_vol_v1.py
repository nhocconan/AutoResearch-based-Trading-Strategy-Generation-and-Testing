#!/usr/bin/env python3
"""
Experiment #231: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with 1d weekly pivot (R1/S1) bias capture institutional flow. Volume spike (>2.0x) filters weak breakouts. ATR stoploss (2.5x) manages risk. Position sizing 0.28 balances return and fee drag. Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets via breakout continuation with pivot bias and in bear markets via mean reversion at opposite band.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_231_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (using Monday of current week as reference)
    # For each 1d bar, calculate pivot based on prior week's high/low/close
    # We'll use the Friday close of prior week as anchor for simplicity
    d_high = df_1d['high'].values.astype(np.float64)
    d_low = df_1d['low'].values.astype(np.float64)
    d_close = df_1d['close'].values.astype(np.float64)
    
    # Weekly high/low/close: resample logic using rolling window of 5 trading days
    # Since we don't have actual week grouping, approximate with 5-day rolling
    wk_high = pd.Series(d_high).rolling(window=5, min_periods=5).max().shift(1).values  # prior week
    wk_low = pd.Series(d_low).rolling(window=5, min_periods=5).min().shift(1).values
    wk_close = pd.Series(d_close).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot calculation: P = (H + L + C) / 3
    weekly_pivot = (wk_high + wk_low + wk_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * weekly_pivot - wk_low
    s1 = 2 * weekly_pivot - wk_high
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 6h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Pivot Bias Conditions ---
        # Long bias: price above weekly pivot and R1 (bullish structure)
        pivot_bias_up = price > weekly_pivot_aligned[i] and price > r1_aligned[i]
        # Short bias: price below weekly pivot and S1 (bearish structure)
        pivot_bias_down = price < weekly_pivot_aligned[i] and price < s1_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if breakout_down and volume_spike and pivot_bias_down:
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
                if breakout_up and volume_spike and pivot_bias_up:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + breakout conditions + pivot bias alignment
        if volume_spike:
            # Long: breakout up AND pivot bias up
            if breakout_up and pivot_bias_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND pivot bias down
            elif breakout_down and pivot_bias_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #231: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with 1d weekly pivot (R1/S1) bias capture institutional flow. Volume spike (>2.0x) filters weak breakouts. ATR stoploss (2.5x) manages risk. Position sizing 0.28 balances return and fee drag. Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets via breakout continuation with pivot bias and in bear markets via mean reversion at opposite band.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_231_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (using Monday of current week as reference)
    # For each 1d bar, calculate pivot based on prior week's high/low/close
    # We'll use the Friday close of prior week as anchor for simplicity
    d_high = df_1d['high'].values.astype(np.float64)
    d_low = df_1d['low'].values.astype(np.float64)
    d_close = df_1d['close'].values.astype(np.float64)
    
    # Weekly high/low/close: resample logic using rolling window of 5 trading days
    # Since we don't have actual week grouping, approximate with 5-day rolling
    wk_high = pd.Series(d_high).rolling(window=5, min_periods=5).max().shift(1).values  # prior week
    wk_low = pd.Series(d_low).rolling(window=5, min_periods=5).min().shift(1).values
    wk_close = pd.Series(d_close).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot calculation: P = (H + L + C) / 3
    weekly_pivot = (wk_high + wk_low + wk_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * weekly_pivot - wk_low
    s1 = 2 * weekly_pivot - wk_high
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 6h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Pivot Bias Conditions ---
        # Long bias: price above weekly pivot and R1 (bullish structure)
        pivot_bias_up = price > weekly_pivot_aligned[i] and price > r1_aligned[i]
        # Short bias: price below weekly pivot and S1 (bearish structure)
        pivot_bias_down = price < weekly_pivot_aligned[i] and price < s1_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if breakout_down and volume_spike and pivot_bias_down:
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
                if breakout_up and volume_spike and pivot_bias_up:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + breakout conditions + pivot bias alignment
        if volume_spike:
            # Long: breakout up AND pivot bias up
            if breakout_up and pivot_bias_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND pivot bias down
            elif breakout_down and pivot_bias_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals