#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12707_6d_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_FAST = 12
EMA_SLOW = 26
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Calculate daily EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = calculate_ema(close_1d, EMA_FAST)
    ema_1d_slow = calculate_ema(close_1d, EMA_SLOW)
    
    # Align daily EMAs to 6h timeframe
    ema_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_fast)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_1d_fast_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]):
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
        
        # Daily EMA trend direction
        daily_uptrend = ema_1d_fast_aligned[i] > ema_1d_slow_aligned[i]
        daily_downtrend = ema_1d_fast_aligned[i] < ema_1d_slow_aligned[i]
        
        # Entry conditions: trend-following with daily filter
        long_entry = daily_uptrend
        short_entry = daily_downtrend
        
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

name = "exp_12707_6d_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_FAST = 12
EMA_SLOW = 26
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Calculate daily EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = calculate_ema(close_1d, EMA_FAST)
    ema_1d_slow = calculate_ema(close_1d, EMA_SLOW)
    
    # Align daily EMAs to 6h timeframe
    ema_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_fast)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_1d_fast_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]):
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
        
        # Daily EMA trend direction
        daily_uptrend = ema_1d_fast_aligned[i] > ema_1d_slow_aligned[i]
        daily_downtrend = ema_1d_fast_aligned[i] < ema_1d_slow_aligned[i]
        
        # Entry conditions: trend-following with daily filter
        long_entry = daily_uptrend
        short_entry = daily_downtrend
        
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