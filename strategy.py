#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Combining 6h Donchian breakouts with 1d/1w weekly pivot levels creates a robust structure-based strategy.
In trending markets (price above/below weekly pivot), we trade breakouts in the direction of the weekly pivot.
In ranging markets (price between weekly pivot levels), we avoid trading to reduce false breakouts.
Volume confirmation (>2.0x average) ensures breakouts have conviction. Uses ATR-based stoploss (2.5x) and 
minimum 6-bar holding period to reduce churn. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_007_6h_donchian20_1d_1w_pivot_vol_v1"
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
    
    # === Calculate weekly pivot from 1d data ===
    # Group by week (starting Monday) and calculate weekly OHLC
    df_1d_copy = df_1d.copy()
    df_1d_copy['week_start'] = pd.to_datetime(df_1d_copy.index).to_period('W').start_time
    weekly = df_1d_copy.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Calculate weekly pivot points (using prior week's data)
    weekly['pivot'] = (weekly['high'].shift(1) + weekly['low'].shift(1) + weekly['close'].shift(1)) / 3
    weekly['r1'] = 2 * weekly['pivot'] - weekly['low'].shift(1)
    weekly['s1'] = 2 * weekly['pivot'] - weekly['high'].shift(1)
    weekly['r2'] = weekly['pivot'] + (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['s2'] = weekly['pivot'] - (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['r3'] = weekly['high'].shift(1) + 2 * (weekly['pivot'] - weekly['low'].shift(1))
    weekly['s3'] = weekly['low'].shift(1) - 2 * (weekly['high'].shift(1) - weekly['pivot'])
    weekly['r4'] = weekly['pivot'] + 3 * (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['s4'] = weekly['pivot'] - 3 * (weekly['high'].shift(1) - weekly['low'].shift(1))
    
    # Forward fill weekly levels to daily data
    weekly_levels = weekly.set_index('week_start')[['pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3', 'r4', 's4']]
    df_1d['week_start'] = pd.to_datetime(df_1d.index).to_period('W').start_time
    df_1d = df_1d.merge(weekly_levels, left_on='week_start', right_index=True, how='left')
    
    # Extract pivot levels as arrays
    pivot_1d = df_1d['pivot'].values
    r1_1d = df_1d['r1'].values
    s1_1d = df_1d['s1'].values
    r2_1d = df_1d['r2'].values
    s2_1d = df_1d['s2'].values
    r3_1d = df_1d['r3'].values
    s3_1d = df_1d['s3'].values
    r4_1d = df_1d['r4'].values
    s4_1d = df_1d['s4'].values
    
    # Align to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
    
    warmup = 200  # Warmup for Donchian and ATR stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Regime Filter ---
        price = close[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Determine market regime based on weekly pivot levels
        is_bullish = price > r1  # Above R1 = bullish bias
        is_bearish = price < s1  # Below S1 = bearish bias
        is_ranging = (price >= s1) and (price <= r1)  # Between S1 and R1 = ranging
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
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
            
            # Minimum holding period of 6 bars to reduce churn (1 day on 6h chart)
            if bars_since_entry < 6:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trade breakouts in direction of weekly pivot bias
        if is_bullish:
            # Long: Donchian breakout up AND volume spike
            if breakout_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif is_bearish:
            # Short: Donchian breakout down AND volume spike
            if breakout_down and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In ranging regime (between S1 and R1), do not trade breakouts
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Combining 6h Donchian breakouts with 1d/1w weekly pivot levels creates a robust structure-based strategy.
In trending markets (price above/below weekly pivot), we trade breakouts in the direction of the weekly pivot.
In ranging markets (price between weekly pivot levels), we avoid trading to reduce false breakouts.
Volume confirmation (>2.0x average) ensures breakouts have conviction. Uses ATR-based stoploss (2.5x) and 
minimum 6-bar holding period to reduce churn. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_007_6h_donchian20_1d_1w_pivot_vol_v1"
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
    
    # === Calculate weekly pivot from 1d data ===
    # Group by week (starting Monday) and calculate weekly OHLC
    df_1d_copy = df_1d.copy()
    df_1d_copy['week_start'] = pd.to_datetime(df_1d_copy.index).to_period('W').start_time
    weekly = df_1d_copy.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Calculate weekly pivot points (using prior week's data)
    weekly['pivot'] = (weekly['high'].shift(1) + weekly['low'].shift(1) + weekly['close'].shift(1)) / 3
    weekly['r1'] = 2 * weekly['pivot'] - weekly['low'].shift(1)
    weekly['s1'] = 2 * weekly['pivot'] - weekly['high'].shift(1)
    weekly['r2'] = weekly['pivot'] + (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['s2'] = weekly['pivot'] - (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['r3'] = weekly['high'].shift(1) + 2 * (weekly['pivot'] - weekly['low'].shift(1))
    weekly['s3'] = weekly['low'].shift(1) - 2 * (weekly['high'].shift(1) - weekly['pivot'])
    weekly['r4'] = weekly['pivot'] + 3 * (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['s4'] = weekly['pivot'] - 3 * (weekly['high'].shift(1) - weekly['low'].shift(1))
    
    # Forward fill weekly levels to daily data
    weekly_levels = weekly.set_index('week_start')[['pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3', 'r4', 's4']]
    df_1d['week_start'] = pd.to_datetime(df_1d.index).to_period('W').start_time
    df_1d = df_1d.merge(weekly_levels, left_on='week_start', right_index=True, how='left')
    
    # Extract pivot levels as arrays
    pivot_1d = df_1d['pivot'].values
    r1_1d = df_1d['r1'].values
    s1_1d = df_1d['s1'].values
    r2_1d = df_1d['r2'].values
    s2_1d = df_1d['s2'].values
    r3_1d = df_1d['r3'].values
    s3_1d = df_1d['s3'].values
    r4_1d = df_1d['r4'].values
    s4_1d = df_1d['s4'].values
    
    # Align to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
    
    warmup = 200  # Warmup for Donchian and ATR stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Regime Filter ---
        price = close[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Determine market regime based on weekly pivot levels
        is_bullish = price > r1  # Above R1 = bullish bias
        is_bearish = price < s1  # Below S1 = bearish bias
        is_ranging = (price >= s1) and (price <= r1)  # Between S1 and R1 = ranging
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
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
            
            # Minimum holding period of 6 bars to reduce churn (1 day on 6h chart)
            if bars_since_entry < 6:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trade breakouts in direction of weekly pivot bias
        if is_bullish:
            # Long: Donchian breakout up AND volume spike
            if breakout_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif is_bearish:
            # Short: Donchian breakout down AND volume spike
            if breakout_down and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In ranging regime (between S1 and R1), do not trade breakouts
            signals[i] = 0.0
    
    return signals