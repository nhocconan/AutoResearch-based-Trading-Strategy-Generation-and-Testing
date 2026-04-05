#!/usr/bin/env python3
"""
Experiment #11234: 1h Momentum with 4h Trend Filter and Volume Spike
Hypothesis: 1h momentum captures short-term moves, filtered by 4h trend to avoid counter-trend trades.
Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Uses 4h trend for direction, 1h only for entry timing to control trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11234_1h_momentum_4h_trend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_momentum(close, period):
    """Calculate momentum (rate of change)"""
    return pd.Series(close).diff(period).values

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

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                    np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, 21)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
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
        
        # Check 4h trend
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Momentum and volume conditions
        mom_up = momentum[i] > 0
        mom_down = momentum[i] < 0
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        adx_strong = adx[i] > ADX_THRESHOLD
        
        # Entry conditions
        long_entry = mom_up and volume_spike and adx_strong and uptrend_4h
        short_entry = mom_down and volume_spike and adx_strong and downtrend_4h
        
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
Experiment #11234: 1h Momentum with 4h Trend Filter and Volume Spike
Hypothesis: 1h momentum captures short-term moves, filtered by 4h trend to avoid counter-trend trades.
Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Uses 4h trend for direction, 1h only for entry timing to control trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11234_1h_momentum_4h_trend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_momentum(close, period):
    """Calculate momentum (rate of change)"""
    return pd.Series(close).diff(period).values

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

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                    np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, 21)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
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
        
        # Check 4h trend
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Momentum and volume conditions
        mom_up = momentum[i] > 0
        mom_down = momentum[i] < 0
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        adx_strong = adx[i] > ADX_THRESHOLD
        
        # Entry conditions
        long_entry = mom_up and volume_spike and adx_strong and uptrend_4h
        short_entry = mom_down and volume_spike and adx_strong and downtrend_4h
        
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
Experiment #11234: 1h Momentum with 4h Trend Filter and Volume Spike
Hypothesis: 1h momentum captures short-term moves, filtered by 4h trend to avoid counter-trend trades.
Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Uses 4h trend for direction, 1h only for entry timing to control trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11234_1h_momentum_4h_trend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_momentum(close, period):
    """Calculate momentum (rate of change)"""
    return pd.Series(close).diff(period).values

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

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                    np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, 21)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
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
        
        # Check 4h trend
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Momentum and volume conditions
        mom_up = momentum[i] > 0
        mom_down = momentum[i] < 0
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        adx_strong = adx[i] > ADX_THRESHOLD
        
        # Entry conditions
        long_entry = mom_up and volume_spike and adx_strong and uptrend_4h
        short_entry = mom_down and volume_spike and adx_strong and downtrend_4h
        
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
Experiment #11234: 1h Momentum with 4h Trend Filter and Volume Spike
Hypothesis: 1h momentum captures short-term moves, filtered by 4h trend to avoid counter-trend trades.
Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Uses 4h trend for direction, 1h only for entry timing to control trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11234_1h_momentum_4h_trend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_momentum(close, period):
    """Calculate momentum (rate of change)"""
    return pd.Series(close).diff(period).values

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

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                    np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, 21)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
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
        
        # Check 4h trend
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Momentum and volume conditions
        mom_up = momentum[i] > 0
        mom_down = momentum[i] < 0
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        adx_strong = adx[i] > ADX_THRESHOLD
        
        # Entry conditions
        long_entry = mom_up and volume_spike and adx_strong and uptrend_4h
        short_entry = mom_down and volume_spike and adx_strong and downtrend_4h
        
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
Experiment #11234: 1h Momentum with 4h Trend Filter and Volume Spike
Hypothesis: 1h momentum captures short-term moves, filtered by 4h trend to avoid counter-trend trades.
Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Uses 4h trend for direction, 1h only for entry timing to control trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11234_1h_momentum_4h_trend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_momentum(close, period):
    """Calculate momentum (rate of change)"""
    return pd.Series(close).diff(period).values

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

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                    np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, 21)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD