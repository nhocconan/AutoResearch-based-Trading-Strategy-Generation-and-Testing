#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11059_6h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 24  # 24 bars of 6h = 6 days
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
CAMARILLA_R3_S3 = 0.25  # Distance to R3/S3 for fade signals
CAMARILLA_R4_S4 = 0.5   # Distance to R4/S4 for breakout signals

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r4 = pivot + (range_ * 1.1 / 2)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    return r3, r4, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
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
    
    # Load daily data for Camarilla calculation ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    r3, r4, s3, s4 = calculate_camarilla(high_d, low_d, close_d)
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_daily, r3)
    r4_6h = align_htf_to_ltf(prices, df_daily, r4)
    s3_6h = align_htf_to_ltf(prices, df_daily, s3)
    s4_6h = align_htf_to_ltf(prices, df_daily, s4)
    
    # Calculate 6h indicators
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    volume_ma = pd.Series(volume_6h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high_6h, low_6h, close_6h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_6h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_6h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Price position relative to Camarilla levels
        price = close_6h[i]
        
        # Fade conditions: price touches R3/S3 with rejection
        near_r3 = abs(price - r3_6h[i]) / price < CAMARILLA_R3_S3
        near_s3 = abs(price - s3_6h[i]) / price < CAMARILLA_R3_S3
        
        # Breakout conditions: price breaks R4/S4 with volume
        breakout_r4 = price > r4_6h[i]
        breakout_s4 = price < s4_6h[i]
        
        # Volume confirmation
        volume_ok = volume_6h[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        fade_long = near_r3 and volume_ok and price < close_6h[i-1]  # Rejection at R3
        fade_short = near_s3 and volume_ok and price > close_6h[i-1]  # Rejection at S3
        breakout_long = breakout_r4 and volume_ok and close_6h[i] > close_6h[i-1]
        breakout_short = breakout_s4 and volume_ok and close_6h[i] < close_6h[i-1]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = price
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = price
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = price
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = price
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

name = "exp_11059_6h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 24  # 24 bars of 6h = 6 days
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
CAMARILLA_R3_S3 = 0.25  # Distance to R3/S3 for fade signals
CAMARILLA_R4_S4 = 0.5   # Distance to R4/S4 for breakout signals

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r4 = pivot + (range_ * 1.1 / 2)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    return r3, r4, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
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
    
    # Load daily data for Camarilla calculation ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    r3, r4, s3, s4 = calculate_camarilla(high_d, low_d, close_d)
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_daily, r3)
    r4_6h = align_htf_to_ltf(prices, df_daily, r4)
    s3_6h = align_htf_to_ltf(prices, df_daily, s3)
    s4_6h = align_htf_to_ltf(prices, df_daily, s4)
    
    # Calculate 6h indicators
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    volume_ma = pd.Series(volume_6h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high_6h, low_6h, close_6h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_6h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_6h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Price position relative to Camarilla levels
        price = close_6h[i]
        
        # Fade conditions: price touches R3/S3 with rejection
        near_r3 = abs(price - r3_6h[i]) / price < CAMARILLA_R3_S3
        near_s3 = abs(price - s3_6h[i]) / price < CAMARILLA_R3_S3
        
        # Breakout conditions: price breaks R4/S4 with volume
        breakout_r4 = price > r4_6h[i]
        breakout_s4 = price < s4_6h[i]
        
        # Volume confirmation
        volume_ok = volume_6h[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        fade_long = near_r3 and volume_ok and price < close_6h[i-1]  # Rejection at R3
        fade_short = near_s3 and volume_ok and price > close_6h[i-1]  # Rejection at S3
        breakout_long = breakout_r4 and volume_ok and close_6h[i] > close_6h[i-1]
        breakout_short = breakout_s4 and volume_ok and close_6h[i] < close_6h[i-1]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = price
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = price
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = price
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = price
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>