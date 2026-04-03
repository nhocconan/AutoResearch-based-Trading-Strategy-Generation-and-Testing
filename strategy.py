#!/usr/bin/env python3
"""
Experiment #219: 6h Donchian(20) Breakout + 12h/1d Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 12h/1d weekly pivot levels capture institutional order flow. 
Weekly pivot direction (from 1d data) filters breakouts to trade with higher-timeframe structure. 
Volume confirmation ensures breakouts have participation. 
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. 
Works in bull/bear by only taking breakouts in direction of 12h/1d pivot trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_pivot_volume_12h_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for weekly pivot calculation ===
    df_12h = get_htf_data(prices, '12h')
    # === HTF: 1d data for pivot reference ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 12h data (approx 1 week = 14x12h bars)
    if len(df_12h) >= 14:
        # Use last 14x12h bars (2 weeks) to calculate pivot for current week
        lookback = min(28, len(df_12h))  # 2 weeks of 12h data
        recent_high = df_12h['high'].iloc[-lookback:].max()
        recent_low = df_12h['low'].iloc[-lookback:].min()
        recent_close = df_12h['close'].iloc[-1]
        pivot_point = (recent_high + recent_low + recent_close) / 3.0
        r1 = 2 * pivot_point - recent_low
        s1 = 2 * pivot_point - recent_high
        r2 = pivot_point + (recent_high - recent_low)
        s2 = pivot_point - (recent_high - recent_low)
        r3 = r2 + (recent_high - recent_low)
        s3 = s2 - (recent_high - recent_low)
        # Pivot trend: bullish if price above pivot, bearish if below
        pivot_trend = np.where(close[:len(df_12h)] > pivot_point, 1, -1)
        # Align to 6h timeframe
        pivot_trend_aligned = align_htf_to_ltf(prices, df_12h, pivot_trend)
    else:
        pivot_trend_aligned = np.full(n, 0)
    
    # Alternative: use 1d data for more stable pivot (weekly from daily)
    if len(df_1d) >= 5:
        # Weekly pivot from last 5 daily bars
        week_high = df_1d['high'].iloc[-5:].max()
        week_low = df_1d['low'].iloc[-5:].min()
        week_close = df_1d['close'].iloc[-1]
        pivot_1d = (week_high + week_low + week_close) / 3.0
        r1_1d = 2 * pivot_1d - week_low
        s1_1d = 2 * pivot_1d - week_high
        r2_1d = pivot_1d + (week_high - week_low)
        s2_1d = pivot_1d - (week_high - week_low)
        # Use 1d pivot as stronger filter
        pivot_trend_1d = np.where(close[:len(df_1d)] > pivot_1d, 1, -1)
        pivot_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_trend_1d)
        # Combine: require both 12h and 1d pivot to agree
        pivot_trend_final = np.where(
            (pivot_trend_1d_aligned != 0) & (pivot_trend_aligned == pivot_trend_1d_aligned),
            pivot_trend_1d_aligned,
            0
        )
    else:
        pivot_trend_final = pivot_trend_aligned if 'pivot_trend_aligned' in locals() else np.zeros(n)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss and volatility filter
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
            np.isnan(vol_ma_20[i]) or i >= len(pivot_trend_final)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from Pivot (12h/1d agreement) ---
        trend_ok_long = pivot_trend_final[i] > 0   # Pivot trend bullish
        trend_ok_short = pivot_trend_final[i] < 0  # Pivot trend bearish
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR pivot turns bearish
                    if close[i] <= dc_lower_20[i] or pivot_trend_final[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR pivot turns bullish
                    if close[i] >= dc_upper_20[i] or pivot_trend_final[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish pivot
        if bullish_breakout and vol_ok and trend_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish pivot
        elif bearish_breakout and vol_ok and trend_ok_short:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #219: 6h Donchian(20) Breakout + 12h/1d Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 12h/1d weekly pivot levels capture institutional order flow. 
Weekly pivot direction (from 1d data) filters breakouts to trade with higher-timeframe structure. 
Volume confirmation ensures breakouts have participation. 
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. 
Works in bull/bear by only taking breakouts in direction of 12h/1d pivot trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_pivot_volume_12h_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for weekly pivot calculation ===
    df_12h = get_htf_data(prices, '12h')
    # === HTF: 1d data for pivot reference ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 12h data (approx 1 week = 14x12h bars)
    if len(df_12h) >= 14:
        # Use last 14x12h bars (2 weeks) to calculate pivot for current week
        lookback = min(28, len(df_12h))  # 2 weeks of 12h data
        recent_high = df_12h['high'].iloc[-lookback:].max()
        recent_low = df_12h['low'].iloc[-lookback:].min()
        recent_close = df_12h['close'].iloc[-1]
        pivot_point = (recent_high + recent_low + recent_close) / 3.0
        r1 = 2 * pivot_point - recent_low
        s1 = 2 * pivot_point - recent_high
        r2 = pivot_point + (recent_high - recent_low)
        s2 = pivot_point - (recent_high - recent_low)
        r3 = r2 + (recent_high - recent_low)
        s3 = s2 - (recent_high - recent_low)
        # Pivot trend: bullish if price above pivot, bearish if below
        pivot_trend = np.where(close[:len(df_12h)] > pivot_point, 1, -1)
        # Align to 6h timeframe
        pivot_trend_aligned = align_htf_to_ltf(prices, df_12h, pivot_trend)
    else:
        pivot_trend_aligned = np.full(n, 0)
    
    # Alternative: use 1d data for more stable pivot (weekly from daily)
    if len(df_1d) >= 5:
        # Weekly pivot from last 5 daily bars
        week_high = df_1d['high'].iloc[-5:].max()
        week_low = df_1d['low'].iloc[-5:].min()
        week_close = df_1d['close'].iloc[-1]
        pivot_1d = (week_high + week_low + week_close) / 3.0
        r1_1d = 2 * pivot_1d - week_low
        s1_1d = 2 * pivot_1d - week_high
        r2_1d = pivot_1d + (week_high - week_low)
        s2_1d = pivot_1d - (week_high - week_low)
        # Use 1d pivot as stronger filter
        pivot_trend_1d = np.where(close[:len(df_1d)] > pivot_1d, 1, -1)
        pivot_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_trend_1d)
        # Combine: require both 12h and 1d pivot to agree
        pivot_trend_final = np.where(
            (pivot_trend_1d_aligned != 0) & (pivot_trend_aligned == pivot_trend_1d_aligned),
            pivot_trend_1d_aligned,
            0
        )
    else:
        pivot_trend_final = pivot_trend_aligned if 'pivot_trend_aligned' in locals() else np.zeros(n)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss and volatility filter
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
            np.isnan(vol_ma_20[i]) or i >= len(pivot_trend_final)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from Pivot (12h/1d agreement) ---
        trend_ok_long = pivot_trend_final[i] > 0   # Pivot trend bullish
        trend_ok_short = pivot_trend_final[i] < 0  # Pivot trend bearish
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR pivot turns bearish
                    if close[i] <= dc_lower_20[i] or pivot_trend_final[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR pivot turns bullish
                    if close[i] >= dc_upper_20[i] or pivot_trend_final[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish pivot
        if bullish_breakout and vol_ok and trend_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish pivot
        elif bearish_breakout and vol_ok and trend_ok_short:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #219: 6h Donchian(20) Breakout + 12h/1d Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 12h/1d weekly pivot levels capture institutional order flow. 
Weekly pivot direction (from 1d data) filters breakouts to trade with higher-timeframe structure. 
Volume confirmation ensures breakouts have participation. 
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. 
Works in bull/bear by only taking breakouts in direction of 12h/1d pivot trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_pivot_volume_12h_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for weekly pivot calculation ===
    df_12h = get_htf_data(prices, '12h')
    # === HTF: 1d data for pivot reference ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 12h data (approx 1 week = 14x12h bars)
    if len(df_12h) >= 14:
        # Use last 14x12h bars (2 weeks) to calculate pivot for current week
        lookback = min(28, len(df_12h))  # 2 weeks of 12h data
        recent_high = df_12h['high'].iloc[-lookback:].max()
        recent_low = df_12h['low'].iloc[-lookback:].min()
        recent_close = df_12h['close'].iloc[-1]
        pivot_point = (recent_high + recent_low + recent_close) / 3.0
        r1 = 2 * pivot_point - recent_low
        s1 = 2 * pivot_point - recent_high
        r2 = pivot_point + (recent_high - recent_low)
        s2 = pivot_point - (recent_high - recent_low)
        r3 = r2 + (recent_high - recent_low)
        s3 = s2 - (recent_high - recent_low)
        # Pivot trend: bullish if price above pivot, bearish if below
        pivot_trend = np.where(close[:len(df_12h)] > pivot_point, 1, -1)
        # Align to 6h timeframe
        pivot_trend_aligned = align_htf_to_ltf(prices, df_12h, pivot_trend)
    else:
        pivot_trend_aligned = np.full(n, 0)
    
    # Alternative: use 1d data for more stable pivot (weekly from daily)
    if len(df_1d) >= 5:
        # Weekly pivot from last 5 daily bars
        week_high = df_1d['high'].iloc[-5:].max()
        week_low = df_1d['low'].iloc[-5:].min()
        week_close = df_1d['close'].iloc[-1]
        pivot_1d = (week_high + week_low + week_close) / 3.0
        r1_1d = 2 * pivot_1d - week_low
        s1_1d = 2 * pivot_1d - week_high
        r2_1d = pivot_1d + (week_high - week_low)
        s2_1d = pivot_1d - (week_high - week_low)
        # Use 1d pivot as stronger filter
        pivot_trend_1d = np.where(close[:len(df_1d)] > pivot_1d, 1, -1)
        pivot_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_trend_1d)
        # Combine: require both 12h and 1d pivot to agree
        pivot_trend_final = np.where(
            (pivot_trend_1d_aligned != 0) & (pivot_trend_aligned == pivot_trend_1d_aligned),
            pivot_trend_1d_aligned,
            0
        )
    else:
        pivot_trend_final = pivot_trend_aligned if 'pivot_trend_aligned' in locals() else np.zeros(n)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss and volatility filter
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
            np.isnan(vol_ma_20[i]) or i >= len(pivot_trend_final)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from Pivot (12h/1d agreement) ---
        trend_ok_long = pivot_trend_final[i] > 0   # Pivot trend bullish
        trend_ok_short = pivot_trend_final[i] < 0  # Pivot trend bearish
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR pivot turns bearish
                    if close[i] <= dc_lower_20[i] or pivot_trend_final[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR pivot turns bullish
                    if close[i] >= dc_upper_20[i] or pivot_trend_final[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish pivot
        if bullish_breakout and vol_ok and trend_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish pivot
        elif bearish_breakout and vol_ok and trend_ok_short:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #219: 6h Donchian(20) Breakout + 12h/1d Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 12h/1d weekly pivot levels capture institutional order flow. 
Weekly pivot direction (from 1d data) filters breakouts to trade with higher-timeframe structure. 
Volume confirmation ensures breakouts have participation. 
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. 
Works in bull/bear by only taking breakouts in direction of 12h/1d pivot trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_pivot_volume_12h_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for weekly pivot calculation ===
    df_12h = get_htf_data(prices, '12h')
    # === HTF: 1d data for pivot reference ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 12h data (approx 1 week = 14x12h bars)
    if len(df_12h) >= 14:
        # Use last 14x12h bars (2 weeks) to calculate pivot for current week
        lookback = min(28, len(df_12h))  # 2 weeks of 12h data
        recent_high = df_12h['high'].iloc[-lookback:].max()
        recent_low = df_12h['low'].iloc[-lookback:].min()
        recent_close = df_12h['close'].iloc[-1]
        pivot_point = (recent_high + recent_low + recent_close) / 3.0
        r1 = 2 * pivot_point - recent_low
        s1 = 2 * pivot_point - recent_high
        r2 = pivot_point + (recent_high - recent_low)
        s2 = pivot_point - (recent_high - recent_low)
        r3 = r2 + (recent_high - recent_low)
        s3 = s2 - (recent_high - recent_low)
        # Pivot trend: bullish if price above pivot, bearish if below
        pivot_trend = np.where(close[:len(df_12h)] > pivot_point, 1, -1)
        # Align to 6h timeframe
        pivot_trend_aligned = align_htf_to_ltf(prices, df_12h, pivot_trend)
    else:
        pivot_trend_aligned = np.full(n, 0)
    
    # Alternative: use 1d data for more stable pivot (weekly from daily)
    if len(df_1d) >= 5:
        # Weekly pivot from last 5 daily bars
        week_high = df_1d['high'].iloc[-5:].max()
        week_low = df_1d['low'].iloc[-5:].min()
        week_close = df_1d['close'].iloc[-1]
        pivot_1d = (week_high + week_low + week_close) / 3.0
        r1_1d = 2 * pivot_1d - week_low
        s1_1d = 2 * pivot_1d - week_high
        r2_1d = pivot_1d + (week_high - week_low)
        s2_1d = pivot_1d - (week_high - week_low)
        # Use 1d pivot as stronger filter
        pivot_trend_1d = np.where(close[:len(df_1d)] > pivot_1d, 1, -1)
        pivot_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_trend_1d)
        # Combine: require both 12h and 1d pivot to agree
        pivot_trend_final = np.where(
            (pivot_trend_1d_aligned != 0) & (pivot_trend_aligned == pivot_trend_1d_aligned),
            pivot_trend_1d_aligned,
            0
        )
    else:
        pivot_trend_final = pivot_trend_aligned if 'pivot_trend_aligned' in locals() else np.zeros(n)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss and volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(