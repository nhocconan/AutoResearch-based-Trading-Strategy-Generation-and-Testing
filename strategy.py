#!/usr/bin/env python3
"""
Experiment #087: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot price action capture institutional swing trades.
Weekly pivot (using prior week's R4/S4 levels) filters for significant structural breaks to avoid false breakouts.
Volume confirmation (1.5x average) ensures participation. Designed for 12-37 trades/year on 6h timeframe.
Uses discrete position sizing (0.30) and ATR-based trailing stop (2.5x) to manage risk in both bull/bear markets.
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
    
    # === HTF: 1d data for weekly pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot points from daily OHLC
    # Prior week's high/low/close for pivot calculation
    week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # Prior week high
    week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values    # Prior week low
    week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(1).values  # Prior week close
    
    # Weekly pivot point (standard calculation)
    pp = (week_high + week_low + week_close) / 3.0
    # Weekly R4 and S4 levels (strong breakout/continuation levels)
    r4 = pp + 3 * (week_high - week_low)  # R4 = PP + 3*(H-L)
    s4 = pp - 3 * (week_high - week_low)  # S4 = PP - 3*(H-L)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed week only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
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
            np.isnan(vol_ma_20[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 6h Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Weekly Pivot Structural Break (R4/S4) ---
        bullish_structural = close[i] > r4_aligned[i]  # Break above weekly R4
        bearish_structural = close[i] < s4_aligned[i]  # Break below weekly S4
        
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
            
            # Exit conditions: price retreats to weekly pivot point or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                # Calculate weekly pivot point for exit (using same logic as entry)
                week_high_exit = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
                week_low_exit = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
                week_close_exit = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(1).values
                pp_exit = (week_high_exit + week_low_exit + week_close_exit) / 3.0
                pp_exit_aligned = align_htf_to_ltf(prices, df_1d, pp_exit)
                
                if position_side > 0:
                    # Exit long: price retreats to weekly PP OR touches lower Donchian
                    if close[i] <= pp_exit_aligned[i] or close[i] <= dc_lower_20[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price rallies to weekly PP OR touches upper Donchian
                    if close[i] >= pp_exit_aligned[i] or close[i] >= dc_upper_20[i]:
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
        # Breakout above upper Donchian with weekly R4 break and volume confirmation
        if bullish_breakout and bullish_structural and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with weekly S4 break and volume confirmation
        elif bearish_breakout and bearish_structural and vol_ok:
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
Experiment #087: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot price action capture institutional swing trades.
Weekly pivot (using prior week's R4/S4 levels) filters for significant structural breaks to avoid false breakouts.
Volume confirmation (1.5x average) ensures participation. Designed for 12-37 trades/year on 6h timeframe.
Uses discrete position sizing (0.30) and ATR-based trailing stop (2.5x) to manage risk in both bull/bear markets.
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
    
    # === HTF: 1d data for weekly pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot points from daily OHLC
    # Prior week's high/low/close for pivot calculation
    week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # Prior week high
    week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values    # Prior week low
    week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(1).values  # Prior week close
    
    # Weekly pivot point (standard calculation)
    pp = (week_high + week_low + week_close) / 3.0
    # Weekly R4 and S4 levels (strong breakout/continuation levels)
    r4 = pp + 3 * (week_high - week_low)  # R4 = PP + 3*(H-L)
    s4 = pp - 3 * (week_high - week_low)  # S4 = PP - 3*(H-L)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed week only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
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
            np.isnan(vol_ma_20[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 6h Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Weekly Pivot Structural Break (R4/S4) ---
        bullish_structural = close[i] > r4_aligned[i]  # Break above weekly R4
        bearish_structural = close[i] < s4_aligned[i]  # Break below weekly S4
        
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
            
            # Exit conditions: price retreats to weekly pivot point or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                # Calculate weekly pivot point for exit (using same logic as entry)
                week_high_exit = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
                week_low_exit = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
                week_close_exit = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(1).values
                pp_exit = (week_high_exit + week_low_exit + week_close_exit) / 3.0
                pp_exit_aligned = align_htf_to_ltf(prices, df_1d, pp_exit)
                
                if position_side > 0:
                    # Exit long: price retreats to weekly PP OR touches lower Donchian
                    if close[i] <= pp_exit_aligned[i] or close[i] <= dc_lower_20[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price rallies to weekly PP OR touches upper Donchian
                    if close[i] >= pp_exit_aligned[i] or close[i] >= dc_upper_20[i]:
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
        # Breakout above upper Donchian with weekly R4 break and volume confirmation
        if bullish_breakout and bullish_structural and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with weekly S4 break and volume confirmation
        elif bearish_breakout and bearish_structural and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>