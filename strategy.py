#!/usr/bin/env python3
"""
Experiment #227: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction (from 1d HTF) capture medium-term momentum with reduced whipsaw. Weekly pivot provides structural support/resistance from higher timeframe, while volume confirmation ensures breakouts have participation. Targets 12-37 trades/year on 6h timeframe. Works in both bull and bear markets by only taking breakouts in the direction of the weekly pivot bias.
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
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 5:
        # Calculate weekly pivot points from prior week's OHLC
        # Weekly high = max of prior week's daily highs
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
        # Weekly low = min of prior week's daily lows
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        # Weekly close = prior week's Friday close (approximated as prior week's last close)
        weekly_close = pd.Series(df_1d['close']).shift(5).values  # 5 trading days ago
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
        weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    else:
        weekly_bias = np.full(n, 0)  # Neutral if insufficient data
    
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
            np.isnan(vol_ma_20[i]) or i >= len(weekly_bias)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from Weekly Pivot Bias ---
        bias_ok_long = weekly_bias[i] > 0   # Weekly bias bullish
        bias_ok_short = weekly_bias[i] < 0  # Weekly bias bearish
        
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
                    # Exit long: price touches lower Donchian OR weekly bias turns bearish
                    if close[i] <= dc_lower_20[i] or weekly_bias[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR weekly bias turns bullish
                    if close[i] >= dc_upper_20[i] or weekly_bias[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish weekly bias
        if bullish_breakout and vol_ok and bias_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish weekly bias
        elif bearish_breakout and vol_ok and bias_ok_short:
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
Experiment #227: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction (from 1d HTF) capture medium-term momentum with reduced whipsaw. Weekly pivot provides structural support/resistance from higher timeframe, while volume confirmation ensures breakouts have participation. Targets 12-37 trades/year on 6h timeframe. Works in both bull and bear markets by only taking breakouts in the direction of the weekly pivot bias.
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
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 5:
        # Calculate weekly pivot points from prior week's OHLC
        # Weekly high = max of prior week's daily highs
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
        # Weekly low = min of prior week's daily lows
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        # Weekly close = prior week's Friday close (approximated as prior week's last close)
        weekly_close = pd.Series(df_1d['close']).shift(5).values  # 5 trading days ago
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
        weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    else:
        weekly_bias = np.full(n, 0)  # Neutral if insufficient data
    
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
            np.isnan(vol_ma_20[i]) or i >= len(weekly_bias)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from Weekly Pivot Bias ---
        bias_ok_long = weekly_bias[i] > 0   # Weekly bias bullish
        bias_ok_short = weekly_bias[i] < 0  # Weekly bias bearish
        
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
                    # Exit long: price touches lower Donchian OR weekly bias turns bearish
                    if close[i] <= dc_lower_20[i] or weekly_bias[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR weekly bias turns bullish
                    if close[i] >= dc_upper_20[i] or weekly_bias[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish weekly bias
        if bullish_breakout and vol_ok and bias_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish weekly bias
        elif bearish_breakout and vol_ok and bias_ok_short:
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
Experiment #227: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction (from 1d HTF) capture medium-term momentum with reduced whipsaw. Weekly pivot provides structural support/resistance from higher timeframe, while volume confirmation ensures breakouts have participation. Targets 12-37 trades/year on 6h timeframe. Works in both bull and bear markets by only taking breakouts in the direction of the weekly pivot bias.
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
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 5:
        # Calculate weekly pivot points from prior week's OHLC
        # Weekly high = max of prior week's daily highs
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
        # Weekly low = min of prior week's daily lows
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        # Weekly close = prior week's Friday close (approximated as prior week's last close)
        weekly_close = pd.Series(df_1d['close']).shift(5).values  # 5 trading days ago
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
        weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    else:
        weekly_bias = np.full(n, 0)  # Neutral if insufficient data
    
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
            np.isnan(vol_ma_20[i]) or i >= len(weekly_bias)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from Weekly Pivot Bias ---
        bias_ok_long = weekly_bias[i] > 0   # Weekly bias bullish
        bias_ok_short = weekly_bias[i] < 0  # Weekly bias bearish
        
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
                    # Exit long: price touches lower Donchian OR weekly bias turns bearish
                    if close[i] <= dc_lower_20[i] or weekly_bias[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR weekly bias turns bullish
                    if close[i] >= dc_upper_20[i] or weekly_bias[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish weekly bias
        if bullish_breakout and vol_ok and bias_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish weekly bias
        elif bearish_breakout and vol_ok and bias_ok_short:
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
Experiment #227: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction (from 1d HTF) capture medium-term momentum with reduced whipsaw. Weekly pivot provides structural support/resistance from higher timeframe, while volume confirmation ensures breakouts have participation. Targets 12-37 trades/year on 6h timeframe. Works in both bull and bear markets by only taking breakouts in the direction of the weekly pivot bias.
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
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 5:
        # Calculate weekly pivot points from prior week's OHLC
        # Weekly high = max of prior week's daily highs
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
        # Weekly low = min of prior week's daily lows
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        # Weekly close = prior week's Friday close (approximated as prior week's last close)
        weekly_close = pd.Series(df_1d['close']).shift(5).values  # 5 trading days ago
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
        weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    else:
        weekly_bias = np.full(n, 0)  # Neutral if insufficient data
    
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
            np.isnan(vol_ma_20[i]) or i >= len(weekly_bias)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from Weekly Pivot Bias ---
        bias_ok_long = weekly_bias[i] > 0   # Weekly bias bullish
        bias_ok_short = weekly_bias[i] < 0  # Weekly bias bearish
        
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
                    # Exit long: price touches lower Donchian OR weekly bias turns bearish
                    if close[i] <= dc_lower_20[i] or weekly_bias[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR weekly bias turns bullish
                    if close[i] >= dc_upper_20[i] or weekly_bias[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish weekly bias
        if bullish_breakout and vol_ok and bias_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish weekly bias
        elif bearish_breakout and vol_ok and bias_ok_short:
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
Experiment #227: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction (from 1d HTF) capture medium-term momentum with reduced whipsaw. Weekly pivot provides structural support/resistance from higher timeframe, while volume confirmation ensures breakouts have participation. Targets 12-37 trades/year on 6h timeframe. Works in both bull and bear markets by only taking breakouts in the direction of the weekly pivot bias.
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
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 5:
        # Calculate weekly pivot points from prior week's OHLC
        # Weekly high = max of prior week's daily highs
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
        # Weekly low = min of prior week's daily lows
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        # Weekly close = prior week's Friday close (approximated as prior week's last close)
        weekly_close = pd.Series(df_1d['close']).shift(5).values  # 5 trading days ago
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
        weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    else:
        weekly_bias = np.full(n, 0)  # Neutral if insufficient data
    
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
            np.isnan(vol_ma_20[i]) or i >= len(weekly_bias)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_