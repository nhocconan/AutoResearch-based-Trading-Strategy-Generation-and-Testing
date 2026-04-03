#!/usr/bin/env python3
"""
Experiment #300: 4h Donchian(20) Breakout + Daily Camarilla Pivot + Volume Spike

HYPOTHESIS: 4h Donchian breakouts aligned with daily Camarilla pivot levels (R4/S4) capture 
strong momentum with institutional participation. Daily pivot provides structural support/resistance, 
reducing false breakouts. Volume confirmation (2.0x average) ensures follow-through. 
Designed for 4h timeframe to target 19-50 trades/year (75-200 over 4 years).
Works in both bull and bear markets by only taking breakouts in direction of daily pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_daily_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels from prior day's OHLC
    if len(df_1d) >= 2:
        # Prior completed day's high, low, close
        daily_high = pd.Series(df_1d['high'].values).shift(1).values
        daily_low = pd.Series(df_1d['low'].values).shift(1).values
        daily_close = pd.Series(df_1d['close'].values).shift(1).values
        
        # Camarilla pivot levels
        daily_range = daily_high - daily_low
        # R4 = Close + Range * 1.1/2
        r4 = daily_close + daily_range * 1.1 / 2
        # S4 = Close - Range * 1.1/2
        s4 = daily_close - daily_range * 1.1 / 2
        
        # Align to 4h timeframe
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
        
        # Daily bias: 1 if price > midpoint (bullish), -1 if price < midpoint (bearish)
        daily_midpoint = (r4_aligned + s4_aligned) / 2
        daily_bias = np.where(close[:len(r4_aligned)] > daily_midpoint, 1, -1)
    else:
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        daily_bias = np.full(n, 0)
    
    # === 4h Indicators ===
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
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
            np.isnan(vol_ma_20[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(daily_bias[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Daily Pivot Levels and Bias ---
        # Only trade breakouts that align with daily bias AND break beyond R4/S4
        bullish_aligned = bullish_breakout and daily_bias[i] > 0 and close[i] > r4_aligned[i]
        bearish_aligned = bearish_breakout and daily_bias[i] < 0 and close[i] < s4_aligned[i]
        
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
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR daily bias turns bearish
                    if close[i] <= dc_lower_20[i] or daily_bias[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR daily bias turns bullish
                    if close[i] >= dc_upper_20[i] or daily_bias[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation, daily bias bullish, and price > R4
        if bullish_aligned and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation, daily bias bearish, and price < S4
        elif bearish_aligned and vol_ok:
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
Experiment #300: 4h Donchian(20) Breakout + Daily Camarilla Pivot + Volume Spike

HYPOTHESIS: 4h Donchian breakouts aligned with daily Camarilla pivot levels (R4/S4) capture 
strong momentum with institutional participation. Daily pivot provides structural support/resistance, 
reducing false breakouts. Volume confirmation (2.0x average) ensures follow-through. 
Designed for 4h timeframe to target 19-50 trades/year (75-200 over 4 years).
Works in both bull and bear markets by only taking breakouts in direction of daily pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_daily_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels from prior day's OHLC
    if len(df_1d) >= 2:
        # Prior completed day's high, low, close
        daily_high = pd.Series(df_1d['high'].values).shift(1).values
        daily_low = pd.Series(df_1d['low'].values).shift(1).values
        daily_close = pd.Series(df_1d['close'].values).shift(1).values
        
        # Camarilla pivot levels
        daily_range = daily_high - daily_low
        # R4 = Close + Range * 1.1/2
        r4 = daily_close + daily_range * 1.1 / 2
        # S4 = Close - Range * 1.1/2
        s4 = daily_close - daily_range * 1.1 / 2
        
        # Align to 4h timeframe
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
        
        # Daily bias: 1 if price > midpoint (bullish), -1 if price < midpoint (bearish)
        daily_midpoint = (r4_aligned + s4_aligned) / 2
        daily_bias = np.where(close[:len(r4_aligned)] > daily_midpoint, 1, -1)
    else:
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        daily_bias = np.full(n, 0)
    
    # === 4h Indicators ===
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
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
            np.isnan(vol_ma_20[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(daily_bias[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Daily Pivot Levels and Bias ---
        # Only trade breakouts that align with daily bias AND break beyond R4/S4
        bullish_aligned = bullish_breakout and daily_bias[i] > 0 and close[i] > r4_aligned[i]
        bearish_aligned = bearish_breakout and daily_bias[i] < 0 and close[i] < s4_aligned[i]
        
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
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR daily bias turns bearish
                    if close[i] <= dc_lower_20[i] or daily_bias[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR daily bias turns bullish
                    if close[i] >= dc_upper_20[i] or daily_bias[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation, daily bias bullish, and price > R4
        if bullish_aligned and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation, daily bias bearish, and price < S4
        elif bearish_aligned and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>