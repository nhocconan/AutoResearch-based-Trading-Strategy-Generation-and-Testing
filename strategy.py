#!/usr/bin/env python3
"""
Experiment #286: 4h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: Combines Donchian breakout with 1d pivot (R1/S1) for directional bias and volume confirmation for strength. Works in bull via breakout continuation and bear via mean reversion at opposite pivot. Discrete sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_286_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot from prior week (5 trading days)
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === 4h Indicators ===
    # Donchian(20)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # For 20-period + 5-day indicators
    
    for i in range(warmup, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume spike filter
        volume_spike = vol_ratio[i] > 1.8
        
        # Donchian breakout
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # Pivot bias
        long_bias = price > r1_aligned[i]
        short_bias = price < s1_aligned[i]
        
        # --- Position Management ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout down with volume + bearish bias
                if breakout_down and volume_spike and short_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout up with volume + bullish bias
                if breakout_up and volume_spike and long_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Entries ---
        if volume_spike:
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                signals[i] = SIZE
            elif breakout_down and short_bias:
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #286: 4h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: Combines Donchian breakout with 1d pivot (R1/S1) for directional bias and volume confirmation for strength. Works in bull via breakout continuation and bear via mean reversion at opposite pivot. Discrete sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_286_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot from prior week (5 trading days)
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === 4h Indicators ===
    # Donchian(20)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # For 20-period + 5-day indicators
    
    for i in range(warmup, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume spike filter
        volume_spike = vol_ratio[i] > 1.8
        
        # Donchian breakout
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # Pivot bias
        long_bias = price > r1_aligned[i]
        short_bias = price < s1_aligned[i]
        
        # --- Position Management ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout down with volume + bearish bias
                if breakout_down and volume_spike and short_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout up with volume + bullish bias
                if breakout_up and volume_spike and long_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Entries ---
        if volume_spike:
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                signals[i] = SIZE
            elif breakout_down and short_bias:
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #286: 4h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: Combines Donchian breakout with 1d pivot (R1/S1) for directional bias and volume confirmation for strength. Works in bull via breakout continuation and bear via mean reversion at opposite pivot. Discrete sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_286_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot from prior week (5 trading days)
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === 4h Indicators ===
    # Donchian(20)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # For 20-period + 5-day indicators
    
    for i in range(warmup, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume spike filter
        volume_spike = vol_ratio[i] > 1.8
        
        # Donchian breakout
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # Pivot bias
        long_bias = price > r1_aligned[i]
        short_bias = price < s1_aligned[i]
        
        # --- Position Management ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout down with volume + bearish bias
                if breakout_down and volume_spike and short_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout up with volume + bullish bias
                if breakout_up and volume_spike and long_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Entries ---
        if volume_spike:
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                signals[i] = SIZE
            elif breakout_down and short_bias:
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #286: 4h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: Combines Donchian breakout with 1d pivot (R1/S1) for directional bias and volume confirmation for strength. Works in bull via breakout continuation and bear via mean reversion at opposite pivot. Discrete sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_286_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot from prior week (5 trading days)
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === 4h Indicators ===
    # Donchian(20)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # For 20-period + 5-day indicators
    
    for i in range(warmup, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume spike filter
        volume_spike = vol_ratio[i] > 1.8
        
        # Donchian breakout
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # Pivot bias
        long_bias = price > r1_aligned[i]
        short_bias = price < s1_aligned[i]
        
        # --- Position Management ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout down with volume + bearish bias
                if breakout_down and volume_spike and short_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout up with volume + bullish bias
                if breakout_up and volume_spike and long_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Entries ---
        if volume_spike:
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                signals[i] = SIZE
            elif breakout_down and short_bias:
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #286: 4h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: Combines Donchian breakout with 1d pivot (R1/S1) for directional bias and volume confirmation for strength. Works in bull via breakout continuation and bear via mean reversion at opposite pivot. Discrete sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_286_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot from prior week (5 trading days)
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === 4h Indicators ===
    # Donchian(20)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # For 20-period + 5-day indicators
    
    for i in range(warmup, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume spike filter
        volume_spike = vol_ratio[i] > 1.8
        
        # Donchian breakout
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # Pivot bias
        long_bias = price > r1_aligned[i]
        short_bias = price < s1_aligned[i]
        
        # --- Position Management ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout down with volume + bearish bias
                if breakout_down and volume_spike and short_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: breakout up with volume + bullish bias
                if breakout_up and volume_spike and long_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Entries ---
        if volume_spike:
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                signals[i] = SIZE
            elif breakout_down and short_bias:
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #286: 4h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: Combines Donchian breakout with 1d pivot (R1/S1) for directional bias and volume confirmation for strength. Works in bull via breakout continuation and bear via mean reversion at opposite pivot. Discrete sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_286_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot levels (Call ONCE before loop) ===
    df_