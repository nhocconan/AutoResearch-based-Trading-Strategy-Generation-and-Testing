#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12567_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1  # Camarilla multiplier for levels
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Typical price for Camarilla calculation
    typical_price = (high + low + close) / 3.0
    # Camarilla levels
    R4 = close + ((high - low) * CAMARILLA_MULTIPLIER * 1.1 / 2)
    R3 = close + ((high - low) * CAMARILLA_MULTIPLIER * 0.55 / 2)
    R2 = close + ((high - low) * CAMARILLA_MULTIPLIER * 0.275 / 2)
    R1 = close + ((high - low) * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S1 = close - ((high - low) * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S2 = close - ((high - low) * CAMARILLA_MULTIPLIER * 0.275 / 2)
    S3 = close - ((high - low) * CAMARILLA_MULTIPLIER * 0.55 / 2)
    S4 = close - ((high - low) * CAMARILLA_MULTIPLIER * 1.1 / 2)
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe (Camarilla levels are constant for the day)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla-based entry conditions
        # Long: price breaks above R3 with volume, target R4
        # Short: price breaks below S3 with volume, target S4
        long_breakout = close[i] > R3_1d_aligned[i] and close[i-1] <= R3_1d_aligned[i-1]
        short_breakout = close[i] < S3_1d_aligned[i] and close[i-1] >= S3_1d_aligned[i-1]
        
        long_entry = volume_ok and long_breakout
        short_entry = volume_ok and short_breakout
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12567_6h_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    R4 = close + (range_hl * CAMARILLA_MULTIPLIER * 1.1 / 2)
    R3 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.55 / 2)
    R2 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.275 / 2)
    R1 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S1 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S2 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.275 / 2)
    S3 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.55 / 2)
    S4 = close - (range_hl * CAMARILLA_MULTIPLIER * 1.1 / 2)
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        if np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Stoploss check
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla breakout with volume
        long_breakout = close[i] > R3_1d_aligned[i] and close[i-1] <= R3_1d_aligned[i-1]
        short_breakout = close[i] < S3_1d_aligned[i] and close[i-1] >= S3_1d_aligned[i-1]
        
        long_entry = volume_ok and long_breakout
        short_entry = volume_ok and short_breakout
        
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12567_6h_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    range_hl = high - low
    R4 = close + (range_hl * CAMARILLA_MULTIPLIER * 1.1 / 2)
    R3 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.55 / 2)
    R2 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.275 / 2)
    R1 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S1 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S2 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.275 / 2)
    S3 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.55 / 2)
    S4 = close - (range_hl * CAMARILLA_MULTIPLIER * 1.1 / 2)
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        if np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Stoploss check
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla breakout with volume
        long_breakout = close[i] > R3_1d_aligned[i] and close[i-1] <= R3_1d_aligned[i-1]
        short_breakout = close[i] < S3_1d_aligned[i] and close[i-1] >= S3_1d_aligned[i-1]
        
        long_entry = volume_ok and long_breakout
        short_entry = volume_ok and short_breakout
        
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12567_6h_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    range_hl = high - low
    R4 = close + (range_hl * CAMARILLA_MULTIPLIER * 1.1 / 2)
    R3 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.55 / 2)
    R2 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.275 / 2)
    R1 = close + (range_hl * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S1 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.091 / 2)
    S2 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.275 / 2)
    S3 = close - (range_hl * CAMARILLA_MULTIPLIER * 0.55 / 2)
    S4 = close - (range_hl * CAMARILLA_MULTIPLIER * 1.1 / 2)
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        if np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Stoploss check
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla breakout with volume
        long_breakout = close[i] > R3_1d_aligned[i] and close[i-1] <= R3_1d_aligned[i-1]
        short_breakout = close[i] < S3_1d_aligned[i] and close[i-1] >= S3_1d_aligned[i-1]
        
        long_entry = volume_ok and long_breakout
        short_entry = volume_ok and short_breakout
        
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

--- END OF FILE ---