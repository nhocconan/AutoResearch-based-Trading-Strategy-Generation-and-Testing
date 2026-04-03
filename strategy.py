#!/usr/bin/env python3
"""
Experiment #258: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Donchian breakouts on daily timeframe capture strong momentum moves. 
Weekly HMA(21) filter ensures alignment with higher timeframe trend to avoid counter-trend trades. 
Volume confirmation filters breakouts with institutional participation. 
ATR-based stoploss manages risk. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) 
to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21_1w = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_1w_padded = np.full(len(close_1w), np.nan)
        hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w[:len(close_1w) - half_len - sqrt_len + 1]
        
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        donchian_high[19:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donchian low (trend reversal)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above Donchian high (trend reversal)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Long: Donchian breakout above upper band + price > weekly HMA (uptrend)
        if (close[i] > donchian_high[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            volume_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakdown below lower band + price < weekly HMA (downtrend)
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_21_1w_aligned[i] and 
              volume_confirmed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #258: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Donchian breakouts on daily timeframe capture strong momentum moves. 
Weekly HMA(21) filter ensures alignment with higher timeframe trend to avoid counter-trend trades. 
Volume confirmation filters breakouts with institutional participation. 
ATR-based stoploss manages risk. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) 
to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21_1w = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_1w_padded = np.full(len(close_1w), np.nan)
        hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w[:len(close_1w) - half_len - sqrt_len + 1]
        
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        donchian_high[19:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donchian low (trend reversal)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above Donchian high (trend reversal)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Long: Donchian breakout above upper band + price > weekly HMA (uptrend)
        if (close[i] > donchian_high[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            volume_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakdown below lower band + price < weekly HMA (downtrend)
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_21_1w_aligned[i] and 
              volume_confirmed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #258: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Donchian breakouts on daily timeframe capture strong momentum moves. 
Weekly HMA(21) filter ensures alignment with higher timeframe trend to avoid counter-trend trades. 
Volume confirmation filters breakouts with institutional participation. 
ATR-based stoploss manages risk. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) 
to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21_1w = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_1w_padded = np.full(len(close_1w), np.nan)
        hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w[:len(close_1w) - half_len - sqrt_len + 1]
        
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        donchian_high[19:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donchian low (trend reversal)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above Donchian high (trend reversal)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Long: Donchian breakout above upper band + price > weekly HMA (uptrend)
        if (close[i] > donchian_high[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            volume_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakdown below lower band + price < weekly HMA (downtrend)
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_21_1w_aligned[i] and 
              volume_confirmed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #258: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Donchian breakouts on daily timeframe capture strong momentum moves. 
Weekly HMA(21) filter ensures alignment with higher timeframe trend to avoid counter-trend trades. 
Volume confirmation filters breakouts with institutional participation. 
ATR-based stoploss manages risk. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) 
to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21_1w = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_1w_padded = np.full(len(close_1w), np.nan)
        hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w[:len(close_1w) - half_len - sqrt_len + 1]
        
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        donchian_high[19:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donchian low (trend reversal)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above Donchian high (trend reversal)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Long: Donchian breakout above upper band + price > weekly HMA (uptrend)
        if (close[i] > donchian_high[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            volume_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakdown below lower band + price < weekly HMA (downtrend)
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_21_1w_aligned[i] and 
              volume_confirmed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #258: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Donchian breakouts on daily timeframe capture strong momentum moves. 
Weekly HMA(21) filter ensures alignment with higher timeframe trend to avoid counter-trend trades. 
Volume confirmation filters breakouts with institutional participation. 
ATR-based stoploss manages risk. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) 
to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21_1w = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_1w_padded = np.full(len(close_1w), np.nan)
        hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w[:len(close_1w) - half_len - sqrt_len + 1]
        
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        donchian_high[19:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donch