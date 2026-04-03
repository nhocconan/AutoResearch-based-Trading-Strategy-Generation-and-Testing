#!/usr/bin/env python3
"""
Experiment #107: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (derived from prior week) capture 
medium-term momentum while avoiding false breakouts. Weekly pivot provides institutional reference levels 
(R3/S3 for fading, R4/S4 for breakout continuation). Volume confirmation (1.5x average) ensures follow-through.
Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
Uses discrete position sizing (0.25) to reduce churn. Works in both bull/bear markets by trading breakouts 
in direction of weekly pivot bias.
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
    # Calculate weekly pivot from prior week's OHLC (requires at least 5 days)
    # We'll use rolling window of 5 days (1 trading week) to compute pivot
    if len(df_1d) < 5:
        # Not enough data, return zeros
        return np.zeros(n)
    
    # Weekly high/low/close from prior 5-day window (shifted by 1 to avoid look-ahead)
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).mean().shift(1).values
    
    # Weekly pivot point (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    r4 = weekly_high + 3 * (weekly_pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - weekly_pivot)
    
    # Align HTF levels to LTF (6h) with shift(1) built-in
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    # ATR for volatility and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel (20-period)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x average)
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
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Bias ---
        # Bias: above weekly pivot = bullish, below = bearish
        pivot_bullish = close[i] > weekly_pivot_aligned[i]
        pivot_bearish = close[i] < weekly_pivot_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: 
            # 1. Minimum hold of 3 bars (~18h)
            # 2. Price reaches opposite weekly pivot level (S3/R3 for fading)
            # 3. Price breaks beyond extreme levels (S4/R4) with reversal
            min_hold = (i - entry_bar) >= 3
            if min_hold:
                if position_side > 0:  # Long position
                    # Exit if price reaches S3 (fade level) or breaks below S4 with reversal
                    if close[i] <= s3_aligned[i] or (close[i] < s4_aligned[i] and close[i] < open[i]):
                        stop_hit = True
                else:  # Short position
                    # Exit if price reaches R3 (fade level) or breaks above R4 with reversal
                    if close[i] >= r3_aligned[i] or (close[i] > r4_aligned[i] and close[i] < open[i]):
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
        # Breakout above upper Donchian with bullish weekly pivot bias and volume confirmation
        if bullish_breakout and pivot_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish weekly pivot bias and volume confirmation
        elif bearish_breakout and pivot_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #107: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (derived from prior week) capture 
medium-term momentum while avoiding false breakouts. Weekly pivot provides institutional reference levels 
(R3/S3 for fading, R4/S4 for breakout continuation). Volume confirmation (1.5x average) ensures follow-through.
Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
Uses discrete position sizing (0.25) to reduce churn. Works in both bull/bear markets by trading breakouts 
in direction of weekly pivot bias.
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
    # Calculate weekly pivot from prior week's OHLC (requires at least 5 days)
    # We'll use rolling window of 5 days (1 trading week) to compute pivot
    if len(df_1d) < 5:
        # Not enough data, return zeros
        return np.zeros(n)
    
    # Weekly high/low/close from prior 5-day window (shifted by 1 to avoid look-ahead)
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).mean().shift(1).values
    
    # Weekly pivot point (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    r4 = weekly_high + 3 * (weekly_pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - weekly_pivot)
    
    # Align HTF levels to LTF (6h) with shift(1) built-in
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    # ATR for volatility and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel (20-period)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x average)
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
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Bias ---
        # Bias: above weekly pivot = bullish, below = bearish
        pivot_bullish = close[i] > weekly_pivot_aligned[i]
        pivot_bearish = close[i] < weekly_pivot_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: 
            # 1. Minimum hold of 3 bars (~18h)
            # 2. Price reaches opposite weekly pivot level (S3/R3 for fading)
            # 3. Price breaks beyond extreme levels (S4/R4) with reversal
            min_hold = (i - entry_bar) >= 3
            if min_hold:
                if position_side > 0:  # Long position
                    # Exit if price reaches S3 (fade level) or breaks below S4 with reversal
                    if close[i] <= s3_aligned[i] or (close[i] < s4_aligned[i] and close[i] < open[i]):
                        stop_hit = True
                else:  # Short position
                    # Exit if price reaches R3 (fade level) or breaks above R4 with reversal
                    if close[i] >= r3_aligned[i] or (close[i] > r4_aligned[i] and close[i] < open[i]):
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
        # Breakout above upper Donchian with bullish weekly pivot bias and volume confirmation
        if bullish_breakout and pivot_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish weekly pivot bias and volume confirmation
        elif bearish_breakout and pivot_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals