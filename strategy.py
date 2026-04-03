#!/usr/bin/env python3
"""
Experiment #099: 6h Williams %R + 12h Supertrend + Volume Spike

HYPOTHESIS: 6h Williams %R identifies overextended moves (oversold < -80, overbought > -20).
12h Supertrend (ATR=10, mult=3) provides trend filter to trade pullbacks in direction of intermediate trend.
Volume spike (>2x 20-period average) confirms institutional participation at turning points.
In bear markets: buy oversold dips in downtrend (counter-trend mean reversion within trend).
In bull markets: sell overbought rallies in uptrend (fade strength into trend).
Discrete position sizing (0.25) and ATR trailing stop (2.5x) manage risk.
Targets 12-30 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_12h_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    n = len(high)
    if n < atr_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First TR
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    supertrend[atr_period-1] = lower_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, n):
        # Upper Band
        if upper_band[i] < supertrend[i-1] or close[i-1] > supertrend[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend[i-1]
            
        # Lower Band
        if lower_band[i] > supertrend[i-1] or close[i-1] < supertrend[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend[i-1]
        
        # Supertrend
        if supertrend[i-1] == upper_band[i-1]:
            if close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values.astype(np.float64)
    low_12h = df_12h['low'].values.astype(np.float64)
    close_12h = df_12h['close'].values.astype(np.float64)
    
    supertrend_12h, supertrend_dir_12h = calculate_supertrend(high_12h, low_12h, close_12h, 10, 3)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    supertrend_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_dir_12h)
    
    # === 6h Indicators ===
    # Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # ATR (14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA (20) for spike confirmation
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
        if (np.isnan(williams_r[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(supertrend_12h_aligned[i]) or np.isnan(supertrend_dir_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Supertrend Trend ---
        trend_bullish = supertrend_dir_12h_aligned[i] > 0
        trend_bearish = supertrend_dir_12h_aligned[i] < 0
        
        # --- Williams %R Conditions ---
        wr_oversold = williams_r[i] < -80  # Oversold
        wr_overbought = williams_r[i] > -20  # Overbought
        
        # --- Volume Confirmation ---
        vol_spike = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2x volume spike
        
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
            
            # Exit conditions: WR normalization or trend change
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: WR returns above -50 OR trend turns bearish
                    if williams_r[i] > -50 or not trend_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: WR returns below -50 OR trend turns bullish
                    if williams_r[i] < -50 or not trend_bearish:
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
        # Williams %R oversold (< -80) with bullish 12h Supertrend trend and volume spike
        if wr_oversold and trend_bullish and vol_spike:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Williams %R overbought (> -20) with bearish 12h Supertrend trend and volume spike
        elif wr_overbought and trend_bearish and vol_spike:
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
Experiment #099: 6h Williams %R + 12h Supertrend + Volume Spike

HYPOTHESIS: 6h Williams %R identifies overextended moves (oversold < -80, overbought > -20).
12h Supertrend (ATR=10, mult=3) provides trend filter to trade pullbacks in direction of intermediate trend.
Volume spike (>2x 20-period average) confirms institutional participation at turning points.
In bear markets: buy oversold dips in downtrend (counter-trend mean reversion within trend).
In bull markets: sell overbought rallies in uptrend (fade strength into trend).
Discrete position sizing (0.25) and ATR trailing stop (2.5x) manage risk.
Targets 12-30 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_12h_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    n = len(high)
    if n < atr_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First TR
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    supertrend[atr_period-1] = lower_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, n):
        # Upper Band
        if upper_band[i] < supertrend[i-1] or close[i-1] > supertrend[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend[i-1]
            
        # Lower Band
        if lower_band[i] > supertrend[i-1] or close[i-1] < supertrend[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend[i-1]
        
        # Supertrend
        if supertrend[i-1] == upper_band[i-1]:
            if close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values.astype(np.float64)
    low_12h = df_12h['low'].values.astype(np.float64)
    close_12h = df_12h['close'].values.astype(np.float64)
    
    supertrend_12h, supertrend_dir_12h = calculate_supertrend(high_12h, low_12h, close_12h, 10, 3)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    supertrend_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_dir_12h)
    
    # === 6h Indicators ===
    # Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # ATR (14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA (20) for spike confirmation
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
        if (np.isnan(williams_r[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(supertrend_12h_aligned[i]) or np.isnan(supertrend_dir_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Supertrend Trend ---
        trend_bullish = supertrend_dir_12h_aligned[i] > 0
        trend_bearish = supertrend_dir_12h_aligned[i] < 0
        
        # --- Williams %R Conditions ---
        wr_oversold = williams_r[i] < -80  # Oversold
        wr_overbought = williams_r[i] > -20  # Overbought
        
        # --- Volume Confirmation ---
        vol_spike = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2x volume spike
        
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
            
            # Exit conditions: WR normalization or trend change
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: WR returns above -50 OR trend turns bearish
                    if williams_r[i] > -50 or not trend_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: WR returns below -50 OR trend turns bullish
                    if williams_r[i] < -50 or not trend_bearish:
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
        # Williams %R oversold (< -80) with bullish 12h Supertrend trend and volume spike
        if wr_oversold and trend_bullish and vol_spike:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Williams %R overbought (> -20) with bearish 12h Supertrend trend and volume spike
        elif wr_overbought and trend_bearish and vol_spike:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals