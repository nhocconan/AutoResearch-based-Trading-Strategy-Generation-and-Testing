#!/usr/bin/env python3
"""
Experiment #4807: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly Camarilla pivot bias (price above/below weekly pivot) with volume confirmation (>1.5x average) capture strong momentum moves while minimizing false breakouts. Uses ATR(14) stoploss (2.0x) for risk control. Designed for 12-37 trades/year on 6h timeframe to avoid fee drag while maintaining statistical significance. Weekly pivot provides structural bias that works in both bull (breakouts above pivot) and bear (breakdowns below pivot) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4807_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: None needed for this strategy ===
    
    # === 1w Indicators: Weekly Camarilla Pivot Levels ===
    if len(df_1w) >= 1:
        # Calculate weekly Camarilla pivot levels from previous week
        # Camarilla: Based on previous week's high, low, close
        prev_high = df_1w['high'].shift(1).values
        prev_low = df_1w['low'].shift(1).values
        prev_close = df_1w['close'].shift(1).values
        
        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Camarilla levels
        range_val = prev_high - prev_low
        r3 = pivot + (range_val * 1.1 / 2)
        r4 = pivot + (range_val * 1.1)
        s3 = pivot - (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1)
        
        # For bias, we use: price > pivot = bullish bias, price < pivot = bearish bias
        weekly_bias = pivot  # Using pivot as the bias reference
    else:
        weekly_bias = np.full(len(df_1w), np.nan)
        r3 = r4 = s3 = s4 = np.full(len(df_1w), np.nan)
    
    # Align HTF weekly bias to 6h timeframe
    if len(weekly_bias) > 0:
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    else:
        weekly_bias_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with weekly pivot bias
        breakout_long = (price >= high_roll[i]) and (price > weekly_bias_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < weekly_bias_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #4807: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly Camarilla pivot bias (price above/below weekly pivot) with volume confirmation (>1.5x average) capture strong momentum moves while minimizing false breakouts. Uses ATR(14) stoploss (2.0x) for risk control. Designed for 12-37 trades/year on 6h timeframe to avoid fee drag while maintaining statistical significance. Weekly pivot provides structural bias that works in both bull (breakouts above pivot) and bear (breakdowns below pivot) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4807_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: None needed for this strategy ===
    
    # === 1w Indicators: Weekly Camarilla Pivot Levels ===
    if len(df_1w) >= 1:
        # Calculate weekly Camarilla pivot levels from previous week
        # Camarilla: Based on previous week's high, low, close
        prev_high = df_1w['high'].shift(1).values
        prev_low = df_1w['low'].shift(1).values
        prev_close = df_1w['close'].shift(1).values
        
        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Camarilla levels
        range_val = prev_high - prev_low
        r3 = pivot + (range_val * 1.1 / 2)
        r4 = pivot + (range_val * 1.1)
        s3 = pivot - (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1)
        
        # For bias, we use: price > pivot = bullish bias, price < pivot = bearish bias
        weekly_bias = pivot  # Using pivot as the bias reference
    else:
        weekly_bias = np.full(len(df_1w), np.nan)
        r3 = r4 = s3 = s4 = np.full(len(df_1w), np.nan)
    
    # Align HTF weekly bias to 6h timeframe
    if len(weekly_bias) > 0:
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    else:
        weekly_bias_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with weekly pivot bias
        breakout_long = (price >= high_roll[i]) and (price > weekly_bias_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < weekly_bias_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals