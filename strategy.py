#!/usr/bin/env python3
"""
Experiment #12314: 1h Volume-Weighted RSI + 4h Trend + 1d Filter
Hypothesis: Use 1d EMA for trend filter, 4h RSI for momentum, and 1h volume-weighted RSI for entry timing.
Volume-weighting filters low-conviction moves. Target: 75-200 total trades over 4 years.
Works in bull (follows 4h/1d trend) and bear (avoids countertrend entries via 1d filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12314_1h_vw_rsi_4h_trend_1d_filter_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
VW_RSI_PERIOD = 14
TREND_EMA_PERIOD = 50
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VW_RSI_THRESHOLD = 50
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vw_rsi(close, volume, period):
    """Calculate Volume-Weighted RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta * volume, 0)
    loss = np.where(delta < 0, -delta * volume, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    vw_rsi = 100 - (100 / (1 + rs))
    return vw_rsi

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, TREND_EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    vw_rsi = calculate_vw_rsi(close, volume, VW_RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VW_RSI_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 4h or 1d EMA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Trend filters (4h and 1d)
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Volume-weighted RSI conditions
        vw_rsi_oversold = vw_rsi[i] < RSI_OVERSOLD
        vw_rsi_overbought = vw_rsi[i] > RSI_OVERBOUGHT
        
        # Entry conditions
        long_entry = vw_rsi_oversold and uptrend_4h and uptrend_1d
        short_entry = vw_rsi_overbought and downtrend_4h and downtrend_1d
        
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
"""
Experiment #12314: 1h Volume-Weighted RSI + 4h Trend + 1d Filter
Hypothesis: Use 1d EMA for trend filter, 4h RSI for momentum, and 1h volume-weighted RSI for entry timing.
Volume-weighting filters low-conviction moves. Target: 75-200 total trades over 4 years.
Works in bull (follows 4h/1d trend) and bear (avoids countertrend entries via 1d filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12314_1h_vw_rsi_4h_trend_1d_filter_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
VW_RSI_PERIOD = 14
TREND_EMA_PERIOD = 50
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VW_RSI_THRESHOLD = 50
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vw_rsi(close, volume, period):
    """Calculate Volume-Weighted RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta * volume, 0)
    loss = np.where(delta < 0, -delta * volume, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    vw_rsi = 100 - (100 / (1 + rs))
    return vw_rsi

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, TREND_EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    vw_rsi = calculate_vw_rsi(close, volume, VW_RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VW_RSI_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 4h or 1d EMA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Trend filters (4h and 1d)
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Volume-weighted RSI conditions
        vw_rsi_oversold = vw_rsi[i] < RSI_OVERSOLD
        vw_rsi_overbought = vw_rsi[i] > RSI_OVERBOUGHT
        
        # Entry conditions
        long_entry = vw_rsi_oversold and uptrend_4h and uptrend_1d
        short_entry = vw_rsi_overbought and downtrend_4h and downtrend_1d
        
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