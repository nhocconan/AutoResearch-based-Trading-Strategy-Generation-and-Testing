#!/usr/bin/env python3
"""
Experiment #262: 12h Donchian Breakout + Daily Volume Spike + Weekly Choppiness Regime

HYPOTHESIS: Combining 12h Donchian(20) breakouts with daily volume confirmation (>1.5x 20-day average) and weekly choppiness filter (CHOP > 61.8 = range, avoid) creates a structured breakout strategy that works in both bull and bear markets. The 12h timeframe minimizes fee drag while capturing multi-day momentum. Volume spike confirms institutional participation. Choppiness regime filter avoids false breakouts in ranging markets. Targets 12-37 trades/year (50-150 total over 4 years) to overcome fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-day average volume on daily data
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    else:
        vol_ma_20_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for choppiness regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index(14) on weekly data
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1w = np.full(len(close_1w), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1w[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 12h timeframe
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel(20)
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    
    for i in range(20-1, n):
        highest_high_20[i] = np.max(high[i-19:i+1])
        lowest_low_20[i] = np.min(low[i-19:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid choppy markets (Choppiness > 61.8 = ranging) ---
        # Only trade when market is trending (Choppiness < 38.2) or moderate (38.2-61.8)
        # Avoid strong ranging regimes where breakouts fail
        if chop_1w_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Current 12h volume > 1.5x 20-day average ---
        # Need to approximate 12h volume from daily - use volume/2 as proxy (12h is half day)
        volume_12h_approx = volume[i] / 2.0
        volume_spike = volume_12h_approx > (1.5 * vol_ma_20_aligned[i])
        
        # --- Donchian Breakout Signals ---
        breakout_up = close[i] > highest_high_20[i-1]  # Break above previous 20-period high
        breakout_down = close[i] < lowest_low_20[i-1]  # Break below previous 20-period low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if high[i] > entry_price + 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if low[i] < entry_price - 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up with volume spike
        if breakout_up and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout down with volume spike
        elif breakout_down and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #262: 12h Donchian Breakout + Daily Volume Spike + Weekly Choppiness Regime

HYPOTHESIS: Combining 12h Donchian(20) breakouts with daily volume confirmation (>1.5x 20-day average) and weekly choppiness filter (CHOP > 61.8 = range, avoid) creates a structured breakout strategy that works in both bull and bear markets. The 12h timeframe minimizes fee drag while capturing multi-day momentum. Volume spike confirms institutional participation. Choppiness regime filter avoids false breakouts in ranging markets. Targets 12-37 trades/year (50-150 total over 4 years) to overcome fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-day average volume on daily data
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    else:
        vol_ma_20_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for choppiness regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index(14) on weekly data
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1w = np.full(len(close_1w), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1w[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 12h timeframe
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel(20)
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    
    for i in range(20-1, n):
        highest_high_20[i] = np.max(high[i-19:i+1])
        lowest_low_20[i] = np.min(low[i-19:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid choppy markets (Choppiness > 61.8 = ranging) ---
        # Only trade when market is trending (Choppiness < 38.2) or moderate (38.2-61.8)
        # Avoid strong ranging regimes where breakouts fail
        if chop_1w_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Current 12h volume > 1.5x 20-day average ---
        # Need to approximate 12h volume from daily - use volume/2 as proxy (12h is half day)
        volume_12h_approx = volume[i] / 2.0
        volume_spike = volume_12h_approx > (1.5 * vol_ma_20_aligned[i])
        
        # --- Donchian Breakout Signals ---
        breakout_up = close[i] > highest_high_20[i-1]  # Break above previous 20-period high
        breakout_down = close[i] < lowest_low_20[i-1]  # Break below previous 20-period low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if high[i] > entry_price + 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if low[i] < entry_price - 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up with volume spike
        if breakout_up and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout down with volume spike
        elif breakout_down and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals