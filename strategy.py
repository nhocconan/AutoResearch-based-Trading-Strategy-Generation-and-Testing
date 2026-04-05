#!/usr/bin/env python3
"""
Experiment #9674: 1h Bollinger Band Breakout with 4h Trend and Volume Confirmation.
Hypothesis: Bollinger Band breakouts on 1h provide high-probability entries when aligned with 4h trend (EMA21) and volume spikes. 
In bull markets, long at upper BB breakout; in bear markets, short at lower BB breakout. 
Volume confirms breakout strength, and 4h EMA21 filter prevents counter-trend trades. 
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9674_1h_bb_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8  # UTC
SESSION_END_HOUR = 20   # UTC

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0  # First value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Check session
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if 4h EMA not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
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
        
        # Breakout conditions
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        
        # Trend filter: only trade in direction of 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = bb_breakout_up and volume_spike and trend_up
        short_entry = bb_breakout_down and volume_spike and trend_down
        
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
Experiment #9674: 1h Bollinger Band Breakout with 4h Trend and Volume Confirmation.
Hypothesis: Bollinger Band breakouts on 1h provide high-probability entries when aligned with 4h trend (EMA21) and volume spikes. 
In bull markets, long at upper BB breakout; in bear markets, short at lower BB breakout. 
Volume confirms breakout strength, and 4h EMA21 filter prevents counter-trend trades. 
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9674_1h_bb_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8  # UTC
SESSION_END_HOUR = 20   # UTC

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0  # First value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Check session
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if 4h EMA not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
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
        
        # Breakout conditions
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        
        # Trend filter: only trade in direction of 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = bb_breakout_up and volume_spike and trend_up
        short_entry = bb_breakout_down and volume_spike and trend_down
        
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
Experiment #9674: 1h Bollinger Band Breakout with 4h Trend and Volume Confirmation.
Hypothesis: Bollinger Band breakouts on 1h provide high-probability entries when aligned with 4h trend (EMA21) and volume spikes. 
In bull markets, long at upper BB breakout; in bear markets, short at lower BB breakout. 
Volume confirms breakout strength, and 4h EMA21 filter prevents counter-trend trades. 
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9674_1h_bb_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8  # UTC
SESSION_END_HOUR = 20   # UTC

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0  # First value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Check session
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if 4h EMA not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
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
        
        # Breakout conditions
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        
        # Trend filter: only trade in direction of 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = bb_breakout_up and volume_spike and trend_up
        short_entry = bb_breakout_down and volume_spike and trend_down
        
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
Experiment #9674: 1h Bollinger Band Breakout with 4h Trend and Volume Confirmation.
Hypothesis: Bollinger Band breakouts on 1h provide high-probability entries when aligned with 4h trend (EMA21) and volume spikes. 
In bull markets, long at upper BB breakout; in bear markets, short at lower BB breakout. 
Volume confirms breakout strength, and 4h EMA21 filter prevents counter-trend trades. 
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9674_1h_bb_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8  # UTC
SESSION_END_HOUR = 20   # UTC

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0  # First value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Check session
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if 4h EMA not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
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
        
        # Breakout conditions
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        
        # Trend filter: only trade in direction of 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = bb_breakout_up and volume_spike and trend_up
        short_entry = bb_breakout_down and volume_spike and trend_down
        
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
Experiment #9674: 1h Bollinger Band Breakout with 4h Trend and Volume Confirmation.
Hypothesis: Bollinger Band breakouts on 1h provide high-probability entries when aligned with 4h trend (EMA21) and volume spikes. 
In bull markets, long at upper BB breakout; in bear markets, short at lower BB breakout. 
Volume confirms breakout strength, and 4h EMA21 filter prevents counter-trend trades. 
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9674_1h_bb_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8  # UTC
SESSION_END_HOUR = 20   # UTC

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0  # First value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Check session
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if 4h EMA not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
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
        
        # Breakout conditions
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        
        # Trend filter: only trade in direction of 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = bb_breakout_up and volume_spike and trend_up
        short_entry = bb_breakout_down and volume_spike and trend_down
        
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
Experiment #9674: 1h Bollinger Band Breakout with 4h Trend and Volume Confirmation.
Hypothesis: Bollinger Band breakouts on 1h provide high-probability entries when aligned with 4h trend (EMA21) and volume spikes. 
In bull markets, long at upper BB breakout; in bear markets, short at lower BB breakout. 
Volume confirms breakout strength, and 4h EMA21 filter prevents counter-trend trades. 
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9674_1h_bb_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8  # UTC
SESSION_END_HOUR = 20   # UTC

def calculate_bollinger_bands(close, period