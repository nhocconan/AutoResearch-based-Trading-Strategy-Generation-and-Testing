# 2025-06-23
#!/usr/bin/env python3
"""
Experiment #9644: 1d Donchian(20) breakout + weekly trend filter + volume confirmation
Hypothesis: Daily Donchian breakouts filtered by weekly trend (price above/below weekly EMA50)
and volume spikes provide high-probability trend continuation. Weekly filter reduces whipsaw
in sideways markets while allowing trends to run. Targets 50-100 trades over 4 years.
Works in bull (breakouts above weekly EMA) and bear (breakdowns below weekly EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9644_1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_EMA_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EMA with alpha=1/period)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=WEEKLY_EMA_PERIOD, adjust=False, min_periods=WEEKLY_EMA_PERIOD).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, WEEKLY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA50
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        bullish_breakout = (donch_upper[i] > 0 and  # Valid Donchian upper
                           close[i] > donch_upper[i] and
                           volume_spike and
                           above_weekly_ema)
                           
        bearish_breakout = (donch_lower[i] > 0 and  # Valid Donchian lower
                           close[i] < donch_lower[i] and
                           volume_spike and
                           below_weekly_ema)
        
        # Generate signals
        if position == 0:
            if bullish_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_breakout:
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
# 2025-06-23
#!/usr/bin/env python3
"""
Experiment #9644: 1d Donchian(20) breakout + weekly trend filter + volume confirmation
Hypothesis: Daily Donchian breakouts filtered by weekly trend (price above/below weekly EMA50)
and volume spikes provide high-probability trend continuation. Weekly filter reduces whipsaw
in sideways markets while allowing trends to run. Targets 50-100 trades over 4 years.
Works in bull (breakouts above weekly EMA) and bear (breakdowns below weekly EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9644_1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_EMA_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EMA with alpha=1/period)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=WEEKLY_EMA_PERIOD, adjust=False, min_periods=WEEKLY_EMA_PERIOD).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, WEEKLY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA50
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        bullish_breakout = (donch_upper[i] > 0 and  # Valid Donchian upper
                           close[i] > donch_upper[i] and
                           volume_spike and
                           above_weekly_ema)
                           
        bearish_breakout = (donch_lower[i] > 0 and  # Valid Donchian lower
                           close[i] < donch_lower[i] and
                           volume_spike and
                           below_weekly_ema)
        
        # Generate signals
        if position == 0:
            if bullish_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_breakout:
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
# 2025-06-23
#!/usr/bin/env python3
"""
Experiment #9644: 1d Donchian(20) breakout + weekly trend filter + volume confirmation
Hypothesis: Daily Donchian breakouts filtered by weekly trend (price above/below weekly EMA50)
and volume spikes provide high-probability trend continuation. Weekly filter reduces whipsaw
in sideways markets while allowing trends to run. Targets 50-100 trades over 4 years.
Works in bull (breakouts above weekly EMA) and bear (breakdowns below weekly EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9644_1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_EMA_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EMA with alpha=1/period)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=WEEKLY_EMA_PERIOD, adjust=False, min_periods=WEEKLY_EMA_PERIOD).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, WEEKLY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA50
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        bullish_breakout = (donch_upper[i] > 0 and  # Valid Donchian upper
                           close[i] > donch_upper[i] and
                           volume_spike and
                           above_weekly_ema)
                           
        bearish_breakout = (donch_lower[i] > 0 and  # Valid Donchian lower
                           close[i] < donch_lower[i] and
                           volume_spike and
                           below_weekly_ema)
        
        # Generate signals
        if position == 0:
            if bullish_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_breakout:
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