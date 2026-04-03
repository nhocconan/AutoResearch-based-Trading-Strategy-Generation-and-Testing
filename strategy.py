#!/usr/bin/env python3
"""
Experiment #159: 6h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 12h HMA trend capture swing momentum
with lower whipsaw than daily timeframe. 12h HMA (21) filters for intermediate trend
direction while being more responsive to regime changes than 1d. Volume confirmation
(1.5x average) ensures institutional participation. Discrete position sizing (0.25)
and ATR trailing stop (2.5x) manage risk. Targets 12-30 trades/year on 6h timeframe
to minimize fee drag. Works in bull/bear markets by trading breakouts in direction
of 12h HMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_hma_12h_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(data, window):
        if len(data) < window:
            return np.full(len(data), np.nan)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(data, weights[::-1], mode='valid') / weights.sum()
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(half) - WMA(full)
    diff = 2 * np.concatenate([np.full(half - 1, np.nan), wma_half]) - np.concatenate([np.full(period - 1, np.nan), wma_full])
    
    # WMA of diff with sqrt_period
    hma = wma(diff, sqrt_period)
    # Adjust for padding
    hma = np.concatenate([np.full(sqrt_period - 1, np.nan), hma])
    
    return hma

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
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
            np.isnan(vol_ma_20[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h HMA Trend ---
        hma_bullish = close[i] > hma_12h_aligned[i]
        hma_bearish = close[i] < hma_12h_aligned[i]
        
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
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below HMA
                    if close[i] <= dc_lower_20[i] or close[i] < hma_12h_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above HMA
                    if close[i] >= dc_upper_20[i] or close[i] > hma_12h_aligned[i]:
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
        # Breakout above upper Donchian with bullish 12h HMA trend and volume confirmation
        if bullish_breakout and hma_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish 12h HMA trend and volume confirmation
        elif bearish_breakout and hma_bearish and vol_ok:
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
Experiment #159: 6h Williams %R Mean Reversion + 12h Supertrend Filter + Volume Spike

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe.
Mean reversion trades are taken only when aligned with 12h Supertrend direction to avoid
fighting the intermediate trend. Volume spike (>2x average) confirms institutional interest
at reversal points. This combination works in both bull (buy oversold in uptrend) and 
sell (sell overbought in downtrend) markets. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Average True Range"""
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Supertrend Indicator"""
    atr = calculate_atr(high, low, close, atr_period)
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
            lowerband[i] = lowerband[i-1]
        if direction[i] == -1 and upperband[i] > upperband[i-1]:
            upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    return supertrend, direction

def calculate_williams_r(high, low, close, period=14):
    """Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    supertrend_12h, supertrend_dir_12h = calculate_supertrend(high_12h, low_12h, close_12h, 10, 3.0)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, supertrend_dir_12h)
    
    # === 6h Indicators ===
    atr_10 = calculate_atr(high, low, close, 10)
    williams_r = calculate_williams_r(high, low, close, 14)
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
        if (np.isnan(atr_10[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(supertrend_dir_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Supertrend Direction ---
        supertrend_bullish = supertrend_dir_aligned[i] == 1
        supertrend_bearish = supertrend_dir_aligned[i] == -1
        
        # --- Williams %R Levels ---
        wr_oversold = williams_r[i] < -80  # Oversold
        wr_overbought = williams_r[i] > -20  # Overbought
        
        # --- Volume Confirmation ---
        vol_spike = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_10[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_10[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: Williams %R reversal or Supertrend flip
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: Williams %R rises above -50 OR Supertrend turns bearish
                    if williams_r[i] > -50 or not supertrend_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: Williams %R falls below -50 OR Supertrend turns bullish
                    if williams_r[i] < -50 or not supertrend_bearish:
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
        # Williams %R oversold with bullish 12h Supertrend and volume spike
        if wr_oversold and supertrend_bullish and vol_spike:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Williams %R overbought with bearish 12h Supertrend and volume spike
        elif wr_overbought and supertrend_bearish and vol_spike:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals