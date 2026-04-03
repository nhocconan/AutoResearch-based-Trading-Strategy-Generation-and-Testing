#!/usr/bin/env python3
"""
Experiment #380: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d HMA trend direction
and confirmed with 4h volume spike, captures strong momentum moves while avoiding false
breakouts in choppy markets. Uses discrete position sizing (0.25) to limit drawdown
during 2022 bear market while maintaining sufficient trades for statistical validity.
Targets 25-50 trades/year (100-200 total over 4 years) to minimize fee drag.
Works in both bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21 = wma(wma_diff, sqrt_len)
        
        # Pad to original length
        hma_21_padded = np.full(len(close_1d), np.nan)
        hma_21_padded[half_len:half_len + len(hma_21)] = hma_21
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) and Volume Ratio ===
    # Donchian(20): highest high and lowest low of past 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio: current vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Calculate ATR(14) for stoploss ---
        if i >= 14:
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
        else:
            atr_14 = 0.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Update highest since entry for trailing stop logic
                highest_since_entry = max(highest_since_entry, high[i])
                # Take profit at 3R (7.5*ATR) or reverse signal
                if close[i] >= entry_price + 7.5 * atr_14:
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
                # Update lowest since entry for trailing stop logic
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Take profit at 3R (7.5*ATR) or reverse signal
                if close[i] <= entry_price - 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price above/below 1d HMA
        price_above_1d_hma = close[i] > hma_21_aligned[i]
        price_below_1d_hma = close[i] < hma_21_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Long: Donchian breakout above highest_20 with volume and uptrend
        long_condition = (
            close[i] > highest_20[i] and 
            volume_spike and 
            price_above_1d_hma
        )
        
        # Short: Donchian breakdown below lowest_20 with volume and downtrend
        short_condition = (
            close[i] < lowest_20[i] and 
            volume_spike and 
            price_below_1d_hma
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #380: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d HMA trend direction
and confirmed with 4h volume spike, captures strong momentum moves while avoiding false
breakouts in choppy markets. Uses discrete position sizing (0.25) to limit drawdown
during 2022 bear market while maintaining sufficient trades for statistical validity.
Targets 25-50 trades/year (100-200 total over 4 years) to minimize fee drag.
Works in both bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21 = wma(wma_diff, sqrt_len)
        
        # Pad to original length
        hma_21_padded = np.full(len(close_1d), np.nan)
        hma_21_padded[half_len:half_len + len(hma_21)] = hma_21
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) and Volume Ratio ===
    # Donchian(20): highest high and lowest low of past 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio: current vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Calculate ATR(14) for stoploss ---
        if i >= 14:
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
        else:
            atr_14 = 0.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Update highest since entry for trailing stop logic
                highest_since_entry = max(highest_since_entry, high[i])
                # Take profit at 3R (7.5*ATR) or reverse signal
                if close[i] >= entry_price + 7.5 * atr_14:
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
                # Update lowest since entry for trailing stop logic
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Take profit at 3R (7.5*ATR) or reverse signal
                if close[i] <= entry_price - 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price above/below 1d HMA
        price_above_1d_hma = close[i] > hma_21_aligned[i]
        price_below_1d_hma = close[i] < hma_21_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Long: Donchian breakout above highest_20 with volume and uptrend
        long_condition = (
            close[i] > highest_20[i] and 
            volume_spike and 
            price_above_1d_hma
        )
        
        # Short: Donchian breakdown below lowest_20 with volume and downtrend
        short_condition = (
            close[i] < lowest_20[i] and 
            volume_spike and 
            price_below_1d_hma
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals