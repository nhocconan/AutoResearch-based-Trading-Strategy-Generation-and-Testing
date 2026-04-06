#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12854_1h_4h_1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
FAST_EMA_PERIOD = 12
SLOW_EMA_PERIOD = 26
SIGNAL_SMOOTH = 9
MACD_THRESHOLD = 0.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
MAX_HOLD_BARS = 48  # Max 48 hours

def calculate_macd(close, fast, slow, smooth):
    """Calculate MACD line and signal line"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=smooth, adjust=False, min_periods=smooth).mean().values
    return macd_line, signal_line

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Calculate 4h EMA trend
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_4h = ema_4h_fast - ema_4h_slow  # Positive = uptrend, negative = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = pd.Series(close_1d).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_1d_slow = pd.Series(close_1d).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_1d = ema_1d_fast - ema_1d_slow  # Positive = uptrend, negative = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    macd_line, macd_signal = calculate_macd(close, FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH)
    macd_hist = macd_line - macd_signal
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if trends not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # MACD histogram crossover with confirmation
        macd_cross_up = macd_hist[i] > MACD_THRESHOLD and macd_hist[i-1] <= MACD_THRESHOLD
        macd_cross_down = macd_hist[i] < MACD_THRESHOLD and macd_hist[i-1] >= MACD_THRESHOLD
        
        # Trend alignment: both 4h and 1d must agree on direction
        trend_up = trend_4h_aligned[i] > 0 and trend_1d_aligned[i] > 0
        trend_down = trend_4h_aligned[i] < 0 and trend_1d_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if macd_cross_up and volume_ok and trend_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif macd_cross_down and volume_ok and trend_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
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

name = "exp_12854_1h_4h_1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
FAST_EMA_PERIOD = 12
SLOW_EMA_PERIOD = 26
SIGNAL_SMOOTH = 9
MACD_THRESHOLD = 0.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
MAX_HOLD_BARS = 48  # Max 48 hours

def calculate_macd(close, fast, slow, smooth):
    """Calculate MACD line and signal line"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=smooth, adjust=False, min_periods=smooth).mean().values
    return macd_line, signal_line

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Calculate 4h EMA trend
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_4h = ema_4h_fast - ema_4h_slow  # Positive = uptrend, negative = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = pd.Series(close_1d).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_1d_slow = pd.Series(close_1d).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_1d = ema_1d_fast - ema_1d_slow  # Positive = uptrend, negative = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    macd_line, macd_signal = calculate_macd(close, FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH)
    macd_hist = macd_line - macd_signal
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if trends not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # MACD histogram crossover with confirmation
        macd_cross_up = macd_hist[i] > MACD_THRESHOLD and macd_hist[i-1] <= MACD_THRESHOLD
        macd_cross_down = macd_hist[i] < MACD_THRESHOLD and macd_hist[i-1] >= MACD_THRESHOLD
        
        # Trend alignment: both 4h and 1d must agree on direction
        trend_up = trend_4h_aligned[i] > 0 and trend_1d_aligned[i] > 0
        trend_down = trend_4h_aligned[i] < 0 and trend_1d_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if macd_cross_up and volume_ok and trend_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif macd_cross_down and volume_ok and trend_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
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

name = "exp_12854_1h_4h_1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
FAST_EMA_PERIOD = 12
SLOW_EMA_PERIOD = 26
SIGNAL_SMOOTH = 9
MACD_THRESHOLD = 0.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
MAX_HOLD_BARS = 48  # Max 48 hours

def calculate_macd(close, fast, slow, smooth):
    """Calculate MACD line and signal line"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=smooth, adjust=False, min_periods=smooth).mean().values
    return macd_line, signal_line

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Calculate 4h EMA trend
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_4h = ema_4h_fast - ema_4h_slow  # Positive = uptrend, negative = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = pd.Series(close_1d).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_1d_slow = pd.Series(close_1d).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_1d = ema_1d_fast - ema_1d_slow  # Positive = uptrend, negative = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    macd_line, macd_signal = calculate_macd(close, FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH)
    macd_hist = macd_line - macd_signal
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if trends not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # MACD histogram crossover with confirmation
        macd_cross_up = macd_hist[i] > MACD_THRESHOLD and macd_hist[i-1] <= MACD_THRESHOLD
        macd_cross_down = macd_hist[i] < MACD_THRESHOLD and macd_hist[i-1] >= MACD_THRESHOLD
        
        # Trend alignment: both 4h and 1d must agree on direction
        trend_up = trend_4h_aligned[i] > 0 and trend_1d_aligned[i] > 0
        trend_down = trend_4h_aligned[i] < 0 and trend_1d_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if macd_cross_up and volume_ok and trend_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif macd_cross_down and volume_ok and trend_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
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

name = "exp_12854_1h_4h_1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
FAST_EMA_PERIOD = 12
SLOW_EMA_PERIOD = 26
SIGNAL_SMOOTH = 9
MACD_THRESHOLD = 0.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
MAX_HOLD_BARS = 48  # Max 48 hours

def calculate_macd(close, fast, slow, smooth):
    """Calculate MACD line and signal line"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=smooth, adjust=False, min_periods=smooth).mean().values
    return macd_line, signal_line

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Calculate 4h EMA trend
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_4h = ema_4h_fast - ema_4h_slow  # Positive = uptrend, negative = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = pd.Series(close_1d).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_1d_slow = pd.Series(close_1d).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_1d = ema_1d_fast - ema_1d_slow  # Positive = uptrend, negative = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    macd_line, macd_signal = calculate_macd(close, FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH)
    macd_hist = macd_line - macd_signal
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(FAST_EMA_PERIOD, SLOW_EMA_PERIOD, SIGNAL_SMOOTH, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if trends not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # MACD histogram crossover with confirmation
        macd_cross_up = macd_hist[i] > MACD_THRESHOLD and macd_hist[i-1] <= MACD_THRESHOLD
        macd_cross_down = macd_hist[i] < MACD_THRESHOLD and macd_hist[i-1] >= MACD_THRESHOLD
        
        # Trend alignment: both 4h and 1d must agree on direction
        trend_up = trend_4h_aligned[i] > 0 and trend_1d_aligned[i] > 0
        trend_down = trend_4h_aligned[i] < 0 and trend_1d_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if macd_cross_up and volume_ok and trend_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif macd_cross_down and volume_ok and trend_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
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

name = "exp_12854_1h_4h_1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
FAST_EMA_PERIOD = 12
SLOW_EMA_PERIOD = 26
SIGNAL_SMOOTH = 9
MACD_THRESHOLD = 0.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
MAX_HOLD_BARS = 48  # Max 48 hours

def calculate_macd(close, fast, slow, smooth):
    """Calculate MACD line and signal line"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=smooth, adjust=False, min_periods=smooth).mean().values
    return macd_line, signal_line

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Calculate 4h EMA trend
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_4h = ema_4h_fast - ema_4h_slow  # Positive = uptrend, negative = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = pd.Series(close_1d).ewm(span=FAST_EMA_PERIOD, adjust=False, min_periods=FAST_EMA_PERIOD).mean().values
    ema_1d_slow = pd.Series(close_1d).ewm(span=SLOW_EMA_PERIOD, adjust=False, min_periods=SLOW_EMA_PERIOD).mean().values
    trend_1d = ema_1d