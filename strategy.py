#!/usr/bin/env python3
"""
Experiment #271: 6h Elder Ray + 1d ADX Trend Filter
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) on 6h identifies momentum strength, filtered by 1d ADX > 25 to ensure trending markets. Long when Bull Power > 0 and Bear Power < 0 with ADX trending; short when Bear Power < 0 and Bull Power > 0 with ADX trending. Uses ATR(6) stoploss. Target: 75-200 total trades over 4 years (19-50/year). Discrete sizing (0.25) minimizes fee drag. Works in bull via trend continuation and bear via strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_271_6h_elder_ray_1d_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # Calculate ADX(14) on 1d
    # True Range
    tr_1d = np.zeros(n_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, n_1d):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    up_move = np.zeros(n_1d)
    down_move = np.zeros(n_1d)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, n_1d):
        up_move[i] = max(high_1d[i] - high_1d[i-1], 0)
        down_move[i] = max(low_1d[i-1] - low_1d[i], 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(up_move).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(down_move).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Align ADX to 6h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    # EMA(13) as the "reference" for power calculation
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # === 6h Indicators: ATR(6) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_6 = pd.Series(tr_6h).ewm(span=6, min_periods=6, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 30  # Enough for EMA(13), ATR(6), and 1d ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_6[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- ADX Trend Filter: Require ADX > 25 for trending market ---
        adx_trending = adx_14_aligned[i] > 25
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_6[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_6[i]
                if high[i] > stop_level:
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
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require ADX trending + Elder Ray conditions
        if adx_trending:
            # Long: Bull Power > 0 AND Bear Power < 0 (strong bullish momentum)
            if bull_power[i] > 0 and bear_power[i] < 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Bear Power < 0 AND Bull Power > 0 (strong bearish momentum)
            elif bear_power[i] < 0 and bull_power[i] > 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
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
Experiment #271: 6h Elder Ray + 1d ADX Trend Filter
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) on 6h identifies momentum strength, filtered by 1d ADX > 25 to ensure trending markets. Long when Bull Power > 0 and Bear Power < 0 with ADX trending; short when Bear Power < 0 and Bull Power > 0 with ADX trending. Uses ATR(6) stoploss. Target: 75-200 total trades over 4 years (19-50/year). Discrete sizing (0.25) minimizes fee drag. Works in bull via trend continuation and bear via strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_271_6h_elder_ray_1d_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # Calculate ADX(14) on 1d
    # True Range
    tr_1d = np.zeros(n_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, n_1d):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    up_move = np.zeros(n_1d)
    down_move = np.zeros(n_1d)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, n_1d):
        up_move[i] = max(high_1d[i] - high_1d[i-1], 0)
        down_move[i] = max(low_1d[i-1] - low_1d[i], 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(up_move).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(down_move).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Align ADX to 6h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    # EMA(13) as the "reference" for power calculation
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # === 6h Indicators: ATR(6) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_6 = pd.Series(tr_6h).ewm(span=6, min_periods=6, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 30  # Enough for EMA(13), ATR(6), and 1d ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_6[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- ADX Trend Filter: Require ADX > 25 for trending market ---
        adx_trending = adx_14_aligned[i] > 25
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_6[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_6[i]
                if high[i] > stop_level:
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
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require ADX trending + Elder Ray conditions
        if adx_trending:
            # Long: Bull Power > 0 AND Bear Power < 0 (strong bullish momentum)
            if bull_power[i] > 0 and bear_power[i] < 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Bear Power < 0 AND Bull Power > 0 (strong bearish momentum)
            elif bear_power[i] < 0 and bull_power[i] > 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals