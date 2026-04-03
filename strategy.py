#!/usr/bin/env python3
"""
Experiment #193: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: 4h Donchian breakouts aligned with 12h HMA trend direction capture strong momentum moves with institutional participation. The 12h HMA provides a smoother trend filter that works in both bull and bear markets by reducing whipsaws. Volume confirmation ensures breakouts have sufficient participation. Targets 19-50 trades/year on 4h timeframe to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 12 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        # Pad to same length
        wma_half_padded = np.full_like(close_12h, np.nan)
        wma_full_padded = np.full_like(close_12h, np.nan)
        wma_half_padded[half_len-1:len(wma_half)+half_len-1] = wma_half
        wma_full_padded[20:len(wma_full)+20] = wma_full
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_21 = wma(raw_hma, sqrt_len)
        # Pad hma_21
        hma_21_padded = np.full_like(close_12h, np.nan)
        hma_21_padded[sqrt_len-1:len(hma_21)+sqrt_len-1] = hma_21
        
        # Align to 4h timeframe
        hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
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
            np.isnan(vol_ma_20[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- 12h HMA Trend Filter ---
        # Need previous and current HMA values to determine slope
        if i >= 1:
            hma_now = hma_21_aligned[i]
            hma_prev = hma_21_aligned[i-1]
            hma_bullish = hma_now > hma_prev  # Rising HMA = uptrend
            hma_bearish = hma_now < hma_prev  # Falling HMA = downtrend
        else:
            hma_bullish = False
            hma_bearish = False
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR HMA turns bearish
                    if close[i] <= dc_lower_20[i] or not hma_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR HMA turns bullish
                    if close[i] >= dc_upper_20[i] or not hma_bearish:
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
        # Breakout above upper Donchian with HMA bullish and volume confirmation
        if bullish_breakout and hma_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with HMA bearish and volume confirmation
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
Experiment #193: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: 4h Donchian breakouts aligned with 12h HMA trend direction capture strong momentum moves with institutional participation. The 12h HMA provides a smoother trend filter that works in both bull and bear markets by reducing whipsaws. Volume confirmation ensures breakouts have sufficient participation. Targets 19-50 trades/year on 4h timeframe to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 12 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        # Pad to same length
        wma_half_padded = np.full_like(close_12h, np.nan)
        wma_full_padded = np.full_like(close_12h, np.nan)
        wma_half_padded[half_len-1:len(wma_half)+half_len-1] = wma_half
        wma_full_padded[20:len(wma_full)+20] = wma_full
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_21 = wma(raw_hma, sqrt_len)
        # Pad hma_21
        hma_21_padded = np.full_like(close_12h, np.nan)
        hma_21_padded[sqrt_len-1:len(hma_21)+sqrt_len-1] = hma_21
        
        # Align to 4h timeframe
        hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
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
            np.isnan(vol_ma_20[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- 12h HMA Trend Filter ---
        # Need previous and current HMA values to determine slope
        if i >= 1:
            hma_now = hma_21_aligned[i]
            hma_prev = hma_21_aligned[i-1]
            hma_bullish = hma_now > hma_prev  # Rising HMA = uptrend
            hma_bearish = hma_now < hma_prev  # Falling HMA = downtrend
        else:
            hma_bullish = False
            hma_bearish = False
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR HMA turns bearish
                    if close[i] <= dc_lower_20[i] or not hma_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR HMA turns bullish
                    if close[i] >= dc_upper_20[i] or not hma_bearish:
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
        # Breakout above upper Donchian with HMA bullish and volume confirmation
        if bullish_breakout and hma_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with HMA bearish and volume confirmation
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