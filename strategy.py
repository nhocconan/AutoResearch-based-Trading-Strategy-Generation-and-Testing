#!/usr/bin/env python3
"""
Experiment #213: 4h Donchian Breakout + 12h HMA Trend + Volume Filter

HYPOTHESIS: 4h Donchian(20) breakouts with volume confirmation, filtered by 12h HMA(21) trend,
generate high-probability trades in both bull and bear markets. The 12h HMA acts as a
smooth trend filter to avoid counter-trend breakouts, while volume confirmation ensures
institutional participation. ATR-based stoploss manages risk. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_213_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h
    def calculate_hma(arr, period):
        """Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA of half period
        wma_half = np.zeros_like(arr)
        for i in range(half_period, len(arr)):
            weights = np.arange(1, half_period + 1)
            wma_half[i] = np.dot(arr[i - half_period + 1:i + 1], weights) / weights.sum()
        
        # WMA of full period
        wma_full = np.zeros_like(arr)
        for i in range(period, len(arr)):
            weights = np.arange(1, period + 1)
            wma_full[i] = np.dot(arr[i - period + 1:i + 1], weights) / weights.sum()
        
        # HMA = 2*WMA(half) - WMA(full)
        hma_raw = 2 * wma_half - wma_full
        
        # Final WMA of sqrt period
        hma = np.zeros_like(arr)
        for i in range(sqrt_period, len(arr)):
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.dot(hma_raw[i - sqrt_period + 1:i + 1], weights) / weights.sum()
        
        return hma
    
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channels (20) ===
    donchian_period = 20
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        dc_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        dc_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 100  # Warmup for 12h indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        price = close[i]
        dc_high_val = dc_high[i]
        dc_low_val = dc_low[i]
        hma_val = hma_12h_aligned[i]
        atr_val = atr_14[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume confirmation
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_val
                if high[i] > stop_level:
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
        # Long breakout: Price > Donchian High + volume spike + price > 12h HMA (uptrend)
        long_breakout = (price > dc_high_val) and vol_spike and (price > hma_val)
        
        # Short breakout: Price < Donchian Low + volume spike + price < 12h HMA (downtrend)
        short_breakout = (price < dc_low_val) and vol_spike and (price < hma_val)
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #213: 4h Donchian Breakout + 12h HMA Trend + Volume Filter

HYPOTHESIS: 4h Donchian(20) breakouts with volume confirmation, filtered by 12h HMA(21) trend,
generate high-probability trades in both bull and bear markets. The 12h HMA acts as a
smooth trend filter to avoid counter-trend breakouts, while volume confirmation ensures
institutional participation. ATR-based stoploss manages risk. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_213_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h
    def calculate_hma(arr, period):
        """Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA of half period
        wma_half = np.zeros_like(arr)
        for i in range(half_period, len(arr)):
            weights = np.arange(1, half_period + 1)
            wma_half[i] = np.dot(arr[i - half_period + 1:i + 1], weights) / weights.sum()
        
        # WMA of full period
        wma_full = np.zeros_like(arr)
        for i in range(period, len(arr)):
            weights = np.arange(1, period + 1)
            wma_full[i] = np.dot(arr[i - period + 1:i + 1], weights) / weights.sum()
        
        # HMA = 2*WMA(half) - WMA(full)
        hma_raw = 2 * wma_half - wma_full
        
        # Final WMA of sqrt period
        hma = np.zeros_like(arr)
        for i in range(sqrt_period, len(arr)):
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.dot(hma_raw[i - sqrt_period + 1:i + 1], weights) / weights.sum()
        
        return hma
    
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channels (20) ===
    donchian_period = 20
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        dc_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        dc_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 100  # Warmup for 12h indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        price = close[i]
        dc_high_val = dc_high[i]
        dc_low_val = dc_low[i]
        hma_val = hma_12h_aligned[i]
        atr_val = atr_14[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume confirmation
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_val
                if high[i] > stop_level:
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
        # Long breakout: Price > Donchian High + volume spike + price > 12h HMA (uptrend)
        long_breakout = (price > dc_high_val) and vol_spike and (price > hma_val)
        
        # Short breakout: Price < Donchian Low + volume spike + price < 12h HMA (downtrend)
        short_breakout = (price < dc_low_val) and vol_spike and (price < hma_val)
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals