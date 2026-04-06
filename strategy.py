#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12527_6d_camarilla1d_v3"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close, period):
    """Calculate Camarilla pivot levels"""
    typical = (high + low + close) / 3
    pivot = typical.rolling(window=period, min_periods=period).mean()
    range_val = high.rolling(window=period, min_periods=period).max() - low.rolling(window=period, min_periods=period).min()
    r3 = pivot + 1.1 * range_val / 2
    s3 = pivot - 1.1 * range_val / 2
    r4 = pivot + 1.1 * range_val
    s4 = pivot - 1.1 * range_val
    return r3.values, s3.values, r4.values, s4.values

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
    r3, s3, r4, s4 = calculate_camarilla(high_1d, low_1d, close_1d, CAMARILLA_PERIOD)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    start = max(CAMARILLA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily Camarilla not available
        if np.isnan(r3_aligned[i]):
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
        
        # Price action relative to Camarilla levels
        at_s3 = np.abs(close[i] - s3_aligned[i]) < (atr[i] * 0.5)  # near S3
        at_r3 = np.abs(close[i] - r3_aligned[i]) < (atr[i] * 0.5)  # near R3
        break_s4 = close[i] > s4_aligned[i]  # break below S4
        break_r4 = close[i] > r4_aligned[i]  # break above R4
        
        # Entry conditions
        long_entry = volume_ok and at_s3 and break_s4  # bounce from S3 with S4 break
        short_entry = volume_ok and at_r3 and break_r4  # rejection at R3 with R4 break
        
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

name = "exp_12527_6d_camarilla1d_v3"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close, period):
    """Calculate Camarilla pivot levels"""
    typical = (high + low + close) / 3
    pivot = typical.rolling(window=period, min_periods=period).mean()
    range_val = high.rolling(window=period, min_periods=period).max() - low.rolling(window=period, min_periods=period).min()
    r3 = pivot + 1.1 * range_val / 2
    s3 = pivot - 1.1 * range_val / 2
    r4 = pivot + 1.1 * range_val
    s4 = pivot - 1.1 * range_val
    return r3.values, s3.values, r4.values, s4.values

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
    r3, s3, r4, s4 = calculate_camarilla(high_1d, low_1d, close_1d, CAMARILLA_PERIOD)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    start = max(CAMARILLA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily Camarilla not available
        if np.isnan(r3_aligned[i]):
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
        
        # Price action relative to Camarilla levels
        at_s3 = np.abs(close[i] - s3_aligned[i]) < (atr[i] * 0.5)  # near S3
        at_r3 = np.abs(close[i] - r3_aligned[i]) < (atr[i] * 0.5)  # near R3
        break_s4 = close[i] > s4_aligned[i]  # break above S4
        break_r4 = close[i] > r4_aligned[i]  # break above R4
        
        # Entry conditions
        long_entry = volume_ok and at_s3 and break_s4  # bounce from S3 with S4 break
        short_entry = volume_ok and at_r3 and break_r4  # rejection at R3 with R4 break
        
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