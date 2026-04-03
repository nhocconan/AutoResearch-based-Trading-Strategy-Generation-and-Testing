#!/usr/bin/env python3
"""
Experiment #235: 6h Williams %R + 12h Supertrend + Volume Spike

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe. 
Entries taken only when aligned with 12h Supertrend direction and confirmed by volume spike (1.8x average). 
Exits on opposite Williams %R extreme or Supertrend reversal. 
Designed for 6h timeframe to capture medium-term reversals in both bull and bear markets. 
Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_12h_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 10:
        # Calculate ATR(10) for Supertrend
        h_12h = df_12h['high'].values
        l_12h = df_12h['low'].values
        c_12h = df_12h['close'].values
        
        tr_12h = np.zeros(len(h_12h))
        tr_12h[0] = h_12h[0] - l_12h[0]
        for i in range(1, len(h_12h)):
            tr_12h[i] = max(h_12h[i] - l_12h[i], abs(h_12h[i] - c_12h[i-1]), abs(l_12h[i] - c_12h[i-1]))
        
        atr_12h = pd.Series(tr_12h).ewm(span=10, min_periods=10, adjust=False).mean().values
        
        # Supertrend calculation
        hl2_12h = (h_12h + l_12h) / 2
        upper_12h = hl2_12h + (3.0 * atr_12h)
        lower_12h = hl2_12h - (3.0 * atr_12h)
        
        supertrend_12h = np.zeros(len(h_12h))
        direction_12h = np.ones(len(h_12h))  # 1 for uptrend, -1 for downtrend
        
        supertrend_12h[0] = upper_12h[0]
        direction_12h[0] = 1
        
        for i in range(1, len(h_12h)):
            if supertrend_12h[i-1] == upper_12h[i-1]:
                supertrend_12h[i] = upper_12h[i] if c_12h[i] <= upper_12h[i-1] else lower_12h[i]
                direction_12h[i] = -1 if c_12h[i] <= upper_12h[i-1] else 1
            else:
                supertrend_12h[i] = lower_12h[i] if c_12h[i] >= lower_12h[i-1] else upper_12h[i]
                direction_12h[i] = 1 if c_12h[i] >= lower_12h[i-1] else -1
        
        # Align to 6h timeframe
        supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
        direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    else:
        supertrend_12h_aligned = np.full(n, 0.0)
        direction_12h_aligned = np.zeros(n)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
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
        if (np.isnan(atr_14[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i]) or 
            i >= len(supertrend_12h_aligned) or i >= len(direction_12h_aligned)):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Conditions ---
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.8 if vol_ma_20[i] > 1e-10 else False  # 1.8x volume spike
        
        # --- Trend Filter from 12h Supertrend Direction ---
        trend_up = direction_12h_aligned[i] > 0   # 12h trend up
        trend_down = direction_12h_aligned[i] < 0  # 12h trend down
        
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
            
            # Exit conditions: Williams %R extreme in opposite direction OR Supertrend reversal
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: overbought Williams %R OR trend turns down
                    if overbought or not trend_up:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: oversold Williams %R OR trend turns up
                    if oversold or not trend_down:
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
        # Williams %R oversold with volume confirmation and bullish 12h trend
        if oversold and vol_ok and trend_up:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Williams %R overbought with volume confirmation and bearish 12h trend
        elif overbought and vol_ok and trend_down:
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
Experiment #235: 6h Williams %R + 12h Supertrend + Volume Spike

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe. 
Entries taken only when aligned with 12h Supertrend direction and confirmed by volume spike (1.8x average). 
Exits on opposite Williams %R extreme or Supertrend reversal. 
Designed for 6h timeframe to capture medium-term reversals in both bull and bear markets. 
Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_12h_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 10:
        # Calculate ATR(10) for Supertrend
        h_12h = df_12h['high'].values
        l_12h = df_12h['low'].values
        c_12h = df_12h['close'].values
        
        tr_12h = np.zeros(len(h_12h))
        tr_12h[0] = h_12h[0] - l_12h[0]
        for i in range(1, len(h_12h)):
            tr_12h[i] = max(h_12h[i] - l_12h[i], abs(h_12h[i] - c_12h[i-1]), abs(l_12h[i] - c_12h[i-1]))
        
        atr_12h = pd.Series(tr_12h).ewm(span=10, min_periods=10, adjust=False).mean().values
        
        # Supertrend calculation
        hl2_12h = (h_12h + l_12h) / 2
        upper_12h = hl2_12h + (3.0 * atr_12h)
        lower_12h = hl2_12h - (3.0 * atr_12h)
        
        supertrend_12h = np.zeros(len(h_12h))
        direction_12h = np.ones(len(h_12h))  # 1 for uptrend, -1 for downtrend
        
        supertrend_12h[0] = upper_12h[0]
        direction_12h[0] = 1
        
        for i in range(1, len(h_12h)):
            if supertrend_12h[i-1] == upper_12h[i-1]:
                supertrend_12h[i] = upper_12h[i] if c_12h[i] <= upper_12h[i-1] else lower_12h[i]
                direction_12h[i] = -1 if c_12h[i] <= upper_12h[i-1] else 1
            else:
                supertrend_12h[i] = lower_12h[i] if c_12h[i] >= lower_12h[i-1] else upper_12h[i]
                direction_12h[i] = 1 if c_12h[i] >= lower_12h[i-1] else -1
        
        # Align to 6h timeframe
        supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
        direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    else:
        supertrend_12h_aligned = np.full(n, 0.0)
        direction_12h_aligned = np.zeros(n)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
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
        if (np.isnan(atr_14[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i]) or 
            i >= len(supertrend_12h_aligned) or i >= len(direction_12h_aligned)):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Conditions ---
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.8 if vol_ma_20[i] > 1e-10 else False  # 1.8x volume spike
        
        # --- Trend Filter from 12h Supertrend Direction ---
        trend_up = direction_12h_aligned[i] > 0   # 12h trend up
        trend_down = direction_12h_aligned[i] < 0  # 12h trend down
        
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
            
            # Exit conditions: Williams %R extreme in opposite direction OR Supertrend reversal
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: overbought Williams %R OR trend turns down
                    if overbought or not trend_up:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: oversold Williams %R OR trend turns up
                    if oversold or not trend_down:
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
        # Williams %R oversold with volume confirmation and bullish 12h trend
        if oversold and vol_ok and trend_up:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Williams %R overbought with volume confirmation and bearish 12h trend
        elif overbought and vol_ok and trend_down:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals