#!/usr/bin/env python3
"""
Experiment #4175: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot bias 
(above weekly pivot = bullish bias, below = bearish bias) and confirmed by volume 
spikes (>1.5x average) capture high-probability trend continuation moves. Weekly 
pivot provides structural support/resistance from higher timeframe, reducing false 
breakouts. Works in both bull/bear via pivot-based directional filter. Targets 
75-150 total trades over 4 years (19-37/year) with 0.25 position size to limit 
drawdown and minimize fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4175_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d for ATR(14) stoploss calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    else:
        atr_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w for pivot points (weekly) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Weekly pivot: (week_high + week_low + week_close) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20 + 5, 20 + 5, 14 + 5)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Donchian breakout conditions
            breakout_upper = price > donch_upper[i]
            breakout_lower = price < donch_lower[i]
            
            # Weekly pivot bias: above pivot = bullish, below = bearish
            bullish_bias = price > weekly_pivot_aligned[i]
            bearish_bias = price < weekly_pivot_aligned[i]
            
            # Long conditions: Donchian upper breakout + bullish pivot bias + volume
            long_entry = breakout_upper and bullish_bias
            
            # Short conditions: Donchian lower breakout + bearish pivot bias + volume
            short_entry = breakout_lower and bearish_bias
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #4175: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot bias 
(above weekly pivot = bullish bias, below = bearish bias) and confirmed by volume 
spikes (>1.5x average) capture high-probability trend continuation moves. Weekly 
pivot provides structural support/resistance from higher timeframe, reducing false 
breakouts. Works in both bull/bear via pivot-based directional filter. Targets 
75-150 total trades over 4 years (19-37/year) with 0.25 position size to limit 
drawdown and minimize fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4175_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d for ATR(14) stoploss calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    else:
        atr_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w for pivot points (weekly) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Weekly pivot: (week_high + week_low + week_close) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20 + 5, 20 + 5, 14 + 5)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Donchian breakout conditions
            breakout_upper = price > donch_upper[i]
            breakout_lower = price < donch_lower[i]
            
            # Weekly pivot bias: above pivot = bullish, below = bearish
            bullish_bias = price > weekly_pivot_aligned[i]
            bearish_bias = price < weekly_pivot_aligned[i]
            
            # Long conditions: Donchian upper breakout + bullish pivot bias + volume
            long_entry = breakout_upper and bullish_bias
            
            # Short conditions: Donchian lower breakout + bearish pivot bias + volume
            short_entry = breakout_lower and bearish_bias
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>