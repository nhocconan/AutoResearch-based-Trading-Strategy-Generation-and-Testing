#!/usr/bin/env python3
"""
Experiment #11319: 6h Williams %R + 1d Trend with Volume Confirmation
Hypothesis: Williams %R identifies overbought/oversold conditions with strong momentum.
In bull markets, buy oversold dips in uptrend; in bear markets, sell overbought rallies in downtrend.
Volume confirms institutional participation. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11319_6h_williamsr_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
WILLIAMS_OVERBOUGHT = -20
WILLIAMS_OVERSOLD = -80
DAILY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr

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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend
    ema_daily = calculate_ema(df_daily['close'].values, DAILY_EMA_PERIOD)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, DAILY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_daily_aligned[i]):
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
        
        # Williams %R conditions
        wr_oversold = williams_r[i] < WILLIAMS_OVERSOLD
        wr_overbought = williams_r[i] > WILLIAMS_OVERBOUGHT
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (daily)
        uptrend_daily = close[i] > ema_daily_aligned[i]
        downtrend_daily = close[i] < ema_daily_aligned[i]
        
        # Entry conditions
        long_entry = wr_oversold and volume_ok and uptrend_daily
        short_entry = wr_overbought and volume_ok and downtrend_daily
        
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
Experiment #11319: 6h Williams %R + 1d Trend with Volume Confirmation
Hypothesis: Williams %R identifies overbought/oversold conditions with strong momentum.
In bull markets, buy oversold dips in uptrend; in bear markets, sell overbought rallies in downtrend.
Volume confirms institutional participation. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11319_6h_williamsr_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
WILLIAMS_OVERBOUGHT = -20
WILLIAMS_OVERSOLD = -80
DAILY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr

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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend
    ema_daily = calculate_ema(df_daily['close'].values, DAILY_EMA_PERIOD)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, DAILY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_daily_aligned[i]):
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
        
        # Williams %R conditions
        wr_oversold = williams_r[i] < WILLIAMS_OVERSOLD
        wr_overbought = williams_r[i] > WILLIAMS_OVERBOUGHT
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (daily)
        uptrend_daily = close[i] > ema_daily_aligned[i]
        downtrend_daily = close[i] < ema_daily_aligned[i]
        
        # Entry conditions
        long_entry = wr_oversold and volume_ok and uptrend_daily
        short_entry = wr_overbought and volume_ok and downtrend_daily
        
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