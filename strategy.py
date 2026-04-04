#!/usr/bin/env python3
"""
Experiment #4947: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1d HTF) with volume confirmation (>1.5x average) capture strong momentum moves in both bull and bear markets. Weekly pivots provide structural support/resistance that price respects, reducing false breakouts. Uses ATR(14) trailing stop (2.0x) for risk management. Targets 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4947_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's H/L/C) ===
    if len(df_1d) >= 5:
        # Calculate weekly high, low, close from daily data
        # We'll use rolling window of 5 days (1 week) to get weekly OHLC
        week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)   # Prior week
        week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1)  # Prior week
        
        # Weekly Pivot Point = (H + L + C) / 3
        pp = (week_high + week_low + week_close) / 3.0
        
        # Weekly Support and Resistance levels
        r1 = 2 * pp - week_low
        s1 = 2 * pp - week_high
        r2 = pp + (week_high - week_low)
        s2 = pp - (week_high - week_low)
        r3 = week_high + 2 * (pp - week_low)
        s3 = week_low - 2 * (week_high - pp)
        
        # Convert to arrays and handle NaN
        pp_arr = pp.values
        r1_arr = r1.values
        s1_arr = s1.values
        r2_arr = r2.values
        s2_arr = s2.values
        r3_arr = r3.values
        s3_arr = s3.values
    else:
        pp_arr = np.full(len(df_1d), np.nan)
        r1_arr = np.full(len(df_1d), np.nan)
        s1_arr = np.full(len(df_1d), np.nan)
        r2_arr = np.full(len(df_1d), np.nan)
        s2_arr = np.full(len(df_1d), np.nan)
        r3_arr = np.full(len(df_1d), np.nan)
        s3_arr = np.full(len(df_1d), np.nan)
    
    # Align HTF weekly pivot levels to 6h timeframe (using prior week's data -> shift(1) already applied)
    if len(pp_arr) > 0:
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp_arr)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1_arr)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1_arr)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2_arr)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2_arr)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3_arr)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3_arr)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
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
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Determine weekly pivot bias: price above PP = bullish bias, below PP = bearish bias
        bullish_bias = price > pp_aligned[i]
        bearish_bias = price < pp_aligned[i]
        
        # Donchian breakout conditions with weekly pivot bias
        breakout_long = (price >= high_roll[i]) and bullish_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_bias and vol_confirm
        
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
Experiment #4947: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1d HTF) with volume confirmation (>1.5x average) capture strong momentum moves in both bull and bear markets. Weekly pivots provide structural support/resistance that price respects, reducing false breakouts. Uses ATR(14) trailing stop (2.0x) for risk management. Targets 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4947_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's H/L/C) ===
    if len(df_1d) >= 5:
        # Calculate weekly high, low, close from daily data
        # We'll use rolling window of 5 days (1 week) to get weekly OHLC
        week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)   # Prior week
        week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1)  # Prior week
        
        # Weekly Pivot Point = (H + L + C) / 3
        pp = (week_high + week_low + week_close) / 3.0
        
        # Weekly Support and Resistance levels
        r1 = 2 * pp - week_low
        s1 = 2 * pp - week_high
        r2 = pp + (week_high - week_low)
        s2 = pp - (week_high - week_low)
        r3 = week_high + 2 * (pp - week_low)
        s3 = week_low - 2 * (week_high - pp)
        
        # Convert to arrays and handle NaN
        pp_arr = pp.values
        r1_arr = r1.values
        s1_arr = s1.values
        r2_arr = r2.values
        s2_arr = s2.values
        r3_arr = r3.values
        s3_arr = s3.values
    else:
        pp_arr = np.full(len(df_1d), np.nan)
        r1_arr = np.full(len(df_1d), np.nan)
        s1_arr = np.full(len(df_1d), np.nan)
        r2_arr = np.full(len(df_1d), np.nan)
        s2_arr = np.full(len(df_1d), np.nan)
        r3_arr = np.full(len(df_1d), np.nan)
        s3_arr = np.full(len(df_1d), np.nan)
    
    # Align HTF weekly pivot levels to 6h timeframe (using prior week's data -> shift(1) already applied)
    if len(pp_arr) > 0:
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp_arr)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1_arr)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1_arr)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2_arr)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2_arr)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3_arr)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3_arr)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
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
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Determine weekly pivot bias: price above PP = bullish bias, below PP = bearish bias
        bullish_bias = price > pp_aligned[i]
        bearish_bias = price < pp_aligned[i]
        
        # Donchian breakout conditions with weekly pivot bias
        breakout_long = (price >= high_roll[i]) and bullish_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_bias and vol_confirm
        
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