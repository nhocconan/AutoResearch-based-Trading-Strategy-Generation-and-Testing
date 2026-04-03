#!/usr/bin/env python3
"""
Experiment #134: 1h Donchian(20) breakout + 4h/1d EMA(50) trend + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d EMA(50) trend and volume confirmation (>1.5x) capture medium-term momentum with controlled trade frequency. Using higher timeframes (4h/1d) for signal direction reduces false breakouts, while 1h provides precise entry timing. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_134_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for EMA(50) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    close_4h = pd.Series(df_4h['close'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    ema_50_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    ema_trend_4h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_trend_1d = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_trend_4h[i]) or
            np.isnan(ema_trend_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Trend Conditions: Require BOTH 4h and 1d EMA(50) alignment ---
        bullish_trend = (close[i] > ema_trend_4h[i]) and (close[i] > ema_trend_1d[i])
        bearish_trend = (close[i] < ema_trend_4h[i]) and (close[i] < ema_trend_1d[i])
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish trend on both HTFs
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish trend on both HTFs
            elif breakout_down and bearish_trend:
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

#!/usr/bin/env python3
"""
Experiment #134: 1h Donchian(20) breakout + 4h/1d EMA(50) trend + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d EMA(50) trend and volume confirmation (>1.5x) capture medium-term momentum with controlled trade frequency. Using higher timeframes (4h/1d) for signal direction reduces false breakouts, while 1h provides precise entry timing. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_134_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for EMA(50) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    close_4h = pd.Series(df_4h['close'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    ema_50_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    ema_trend_4h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_trend_1d = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_trend_4h[i]) or
            np.isnan(ema_trend_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Trend Conditions: Require BOTH 4h and 1d EMA(50) alignment ---
        bullish_trend = (close[i] > ema_trend_4h[i]) and (close[i] > ema_trend_1d[i])
        bearish_trend = (close[i] < ema_trend_4h[i]) and (close[i] < ema_trend_1d[i])
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish trend on both HTFs
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish trend on both HTFs
            elif breakout_down and bearish_trend:
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

#!/usr/bin/env python3
"""
Experiment #134: 1h Donchian(20) breakout + 4h/1d EMA(50) trend + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d EMA(50) trend and volume confirmation (>1.5x) capture medium-term momentum with controlled trade frequency. Using higher timeframes (4h/1d) for signal direction reduces false breakouts, while 1h provides precise entry timing. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_134_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for EMA(50) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    close_4h = pd.Series(df_4h['close'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    ema_50_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    ema_trend_4h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_trend_1d = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_trend_4h[i]) or
            np.isnan(ema_trend_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Trend Conditions: Require BOTH 4h and 1d EMA(50) alignment ---
        bullish_trend = (close[i] > ema_trend_4h[i]) and (close[i] > ema_trend_1d[i])
        bearish_trend = (close[i] < ema_trend_4h[i]) and (close[i] < ema_trend_1d[i])
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish trend on both HTFs
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish trend on both HTFs
            elif breakout_down and bearish_trend:
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

#!/usr/bin/env python3
"""
Experiment #134: 1h Donchian(20) breakout + 4h/1d EMA(50) trend + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d EMA(50) trend and volume confirmation (>1.5x) capture medium-term momentum with controlled trade frequency. Using higher timeframes (4h/1d) for signal direction reduces false breakouts, while 1h provides precise entry timing. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_134_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for EMA(50) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    close_4h = pd.Series(df_4h['close'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    ema_50_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    ema_trend_4h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_trend_1d = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_trend_4h[i]) or
            np.isnan(ema_trend_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Trend Conditions: Require BOTH 4h and 1d EMA(50) alignment ---
        bullish_trend = (close[i] > ema_trend_4h[i]) and (close[i] > ema_trend_1d[i])
        bearish_trend = (close[i] < ema_trend_4h[i]) and (close[i] < ema_trend_1d[i])
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish trend on both HTFs
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish trend on both HTFs
            elif breakout_down and bearish_trend:
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

#!/usr/bin/env python3
"""
Experiment #134: 1h Donchian(20) breakout + 4h/1d EMA(50) trend + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d EMA(50) trend and volume confirmation (>1.5x) capture medium-term momentum with controlled trade frequency. Using higher timeframes (4h/1d) for signal direction reduces false breakouts, while 1h provides precise entry timing. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_134_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for EMA(50) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    close_4h = pd.Series(df_4h['close'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    ema_50_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    ema_trend_4h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_trend_1d = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_trend_4h[i]) or
            np.isnan(ema_trend_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0