#!/usr/bin/env python3
"""
Experiment #291: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Combining 6h Donchian(20) breakouts with 1d weekly pivot levels (using Friday's close as weekly anchor) and volume confirmation creates a robust breakout strategy. The weekly pivot provides institutional reference points, Donchian captures breakouts from consolidation, and volume filters out fakeouts. Works in both bull (continuation breakouts) and bear (breakdowns) markets. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while capturing high-probability breakouts with institutional participation.
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
    
    # === HTF: 1d data for weekly pivot and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points using Friday's close as anchor
    # Weekly pivot = (Friday High + Friday Low + Friday Close) / 3
    # We'll use the most recent Friday's data available
    if len(df_1d) >= 1:
        # Calculate typical price for each day
        typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
        typical_price_1d_values = typical_price_1d.values
        
        # For weekly pivot, we use the last Friday's typical price
        # Since we don't have day-of-week easily, we'll use a rolling approach
        # that captures weekly levels: use 5-day lookback for weekly approximation
        weekly_pivot = pd.Series(typical_price_1d_values).rolling(window=5, min_periods=5).mean().values
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        
        # Also calculate weekly support/resistance levels
        weekly_range = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=5, min_periods=5).mean().values
        weekly_r1 = weekly_pivot + weekly_range  # Weekly R1
        weekly_s1 = weekly_pivot - weekly_range  # Weekly S1
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Donchian Channel (20-period)
    donchian_window = 20
    if n >= donchian_window:
        # Upper band: highest high over donchian_window periods
        donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
        # Lower band: lowest low over donchian_window periods
        donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = max(100, donchian_window)  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: Reverse on opposite signal or stoploss ---
        if in_position:
            # Calculate ATR-based stoploss (2*ATR)
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.0 * atr_14
                # Exit conditions: stoploss hit OR reverse signal (breakdown below weekly S1 with volume)
                if low[i] < stop_level or (close[i] < weekly_s1_aligned[i] and volume_spike[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.0 * atr_14
                # Exit conditions: stoploss hit OR reverse signal (breakout above weekly R1 with volume)
                if high[i] > stop_level or (close[i] > weekly_r1_aligned[i] and volume_spike[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Breakout above Donchian high with volume confirmation AND price above weekly pivot (bullish bias)
        if (close[i] > donchian_high[i] and volume_spike[i] and 
            close[i] > weekly_pivot_aligned[i]):
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Breakdown below Donchian low with volume confirmation AND price below weekly pivot (bearish bias)
        elif (close[i] < donchian_low[i] and volume_spike[i] and 
              close[i] < weekly_pivot_aligned[i]):
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #291: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Combining 6h Donchian(20) breakouts with 1d weekly pivot levels (using Friday's close as weekly anchor) and volume confirmation creates a robust breakout strategy. The weekly pivot provides institutional reference points, Donchian captures breakouts from consolidation, and volume filters out fakeouts. Works in both bull (continuation breakouts) and bear (breakdowns) markets. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while capturing high-probability breakouts with institutional participation.
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
    
    # === HTF: 1d data for weekly pivot and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points using Friday's close as anchor
    # Weekly pivot = (Friday High + Friday Low + Friday Close) / 3
    # We'll use the most recent Friday's data available
    if len(df_1d) >= 1:
        # Calculate typical price for each day
        typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
        typical_price_1d_values = typical_price_1d.values
        
        # For weekly pivot, we use the last Friday's typical price
        # Since we don't have day-of-week easily, we'll use a rolling approach
        # that captures weekly levels: use 5-day lookback for weekly approximation
        weekly_pivot = pd.Series(typical_price_1d_values).rolling(window=5, min_periods=5).mean().values
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        
        # Also calculate weekly support/resistance levels
        weekly_range = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=5, min_periods=5).mean().values
        weekly_r1 = weekly_pivot + weekly_range  # Weekly R1
        weekly_s1 = weekly_pivot - weekly_range  # Weekly S1
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Donchian Channel (20-period)
    donchian_window = 20
    if n >= donchian_window:
        # Upper band: highest high over donchian_window periods
        donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
        # Lower band: lowest low over donchian_window periods
        donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = max(100, donchian_window)  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: Reverse on opposite signal or stoploss ---
        if in_position:
            # Calculate ATR-based stoploss (2*ATR)
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.0 * atr_14
                # Exit conditions: stoploss hit OR reverse signal (breakdown below weekly S1 with volume)
                if low[i] < stop_level or (close[i] < weekly_s1_aligned[i] and volume_spike[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.0 * atr_14
                # Exit conditions: stoploss hit OR reverse signal (breakout above weekly R1 with volume)
                if high[i] > stop_level or (close[i] > weekly_r1_aligned[i] and volume_spike[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Breakout above Donchian high with volume confirmation AND price above weekly pivot (bullish bias)
        if (close[i] > donchian_high[i] and volume_spike[i] and 
            close[i] > weekly_pivot_aligned[i]):
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Breakdown below Donchian low with volume confirmation AND price below weekly pivot (bearish bias)
        elif (close[i] < donchian_low[i] and volume_spike[i] and 
              close[i] < weekly_pivot_aligned[i]):
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals