#!/usr/bin/env python3
"""
Experiment #4111: 6h Donchian(20) breakout + 1d volume spike + ADX(14) regime filter
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>2x average volume) and ADX>25 (trending regime) capture strong momentum moves while avoiding choppy markets. Volume spike ensures institutional participation, ADX filter avoids whipsaws in ranging markets. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4111_6h_donchian20_1d_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d ADX(14) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # Calculate ADX components
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False).mean().values
        plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
        minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di_14 = 100 * plus_dm_14 / tr_14
        minus_di_14 = 100 * minus_dm_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
        adx_14 = pd.Series(dx).ewm(span=14, adjust=False).mean().values
        adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    else:
        adx_14_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 14)  # DC lookback, vol MA and ADX buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) and ADX > 25 (trending regime)
        volume_spike = vol_ratio[i] > 2.0
        strong_trend = adx_14_aligned[i] > 25.0
        
        if volume_spike and strong_trend:
            # Breakout logic: 
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long entry: upper Donchian breakout
            long_entry = breakout_up
            
            # Short entry: lower Donchian breakout
            short_entry = breakout_down
            
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
Experiment #4111: 6h Donchian(20) breakout + 1d volume spike + ADX(14) regime filter
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>2x average volume) and ADX>25 (trending regime) capture strong momentum moves while avoiding choppy markets. Volume spike ensures institutional participation, ADX filter avoids whipsaws in ranging markets. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4111_6h_donchian20_1d_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d ADX(14) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # Calculate ADX components
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False).mean().values
        plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
        minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di_14 = 100 * plus_dm_14 / tr_14
        minus_di_14 = 100 * minus_dm_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
        adx_14 = pd.Series(dx).ewm(span=14, adjust=False).mean().values
        adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    else:
        adx_14_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 14)  # DC lookback, vol MA and ADX buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) and ADX > 25 (trending regime)
        volume_spike = vol_ratio[i] > 2.0
        strong_trend = adx_14_aligned[i] > 25.0
        
        if volume_spike and strong_trend:
            # Breakout logic: 
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long entry: upper Donchian breakout
            long_entry = breakout_up
            
            # Short entry: lower Donchian breakout
            short_entry = breakout_down
            
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