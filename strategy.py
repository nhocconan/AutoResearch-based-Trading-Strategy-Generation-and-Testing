#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and 1-hour EMA trend filter.
# Donchian breakouts capture momentum bursts; volume confirms institutional participation;
# EMA ensures alignment with short-term trend to avoid whipsaws. Works in bull (breakouts above) and bear (breakdowns below).
# Target: 80-180 total trades over 4 years (20-45/year) to stay within optimal range.

name = "exp_13326_4h_donchian20_1h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1-hour data ONCE before loop for EMA and volume
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1-hour EMA for trend filter
    close_1h = df_1h['close'].values
    ema_1h = calculate_ema(close_1h, EMA_PERIOD)
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    # Calculate 1-hour volume MA for confirmation
    volume_1h = df_1h['volume'].values
    volume_ma_1h = pd.Series(volume_1h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    
    # Calculate ATR for stoploss
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
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(ema_1h_aligned[i]) or np.isnan(volume_ma_1h_aligned[i]):
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
        
        # Volume confirmation: current 4h volume > 1.5x 1h volume MA
        # Note: 1h volume MA is aligned to 4h, so we compare 4h volume to 1h MA
        volume_ok = volume[i] > (volume_ma_1h_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1h_aligned[i]) else False
        
        # Trend filter: price above/below 1h EMA
        uptrend = close[i] > ema_1h_aligned[i]
        downtrend = close[i] < ema_1h_aligned[i]
        
        # Donchian breakout levels (using previous period's high/low to avoid look-ahead)
        # We use the highest high and lowest low of the past DONCHIAN_PERIOD periods
        # For 4h timeframe, we look back DONCHIAN_PERIOD candles
        if i >= DONCHIAN_PERIOD:
            donchian_high = np.max(high[i-DONCHIAN_PERIOD:i])
            donchian_low = np.min(low[i-DONCHIAN_PERIOD:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        # Breakout signals
        breakout_up = volume_ok and uptrend and (not np.isnan(donchian_high)) and (high[i] > donchian_high)
        breakout_down = volume_ok and downtrend and (not np.isnan(donchian_low)) and (low[i] < donchian_low)
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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

# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and 1-hour EMA trend filter.
# Donchian breakouts capture momentum bursts; volume confirms institutional participation;
# EMA ensures alignment with short-term trend to avoid whipsaws. Works in bull (breakouts above) and bear (breakdowns below).
# Target: 80-180 total trades over 4 years (20-45/year) to stay within optimal range.

name = "exp_13326_4h_donchian20_1h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1-hour data ONCE before loop for EMA and volume
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1-hour EMA for trend filter
    close_1h = df_1h['close'].values
    ema_1h = calculate_ema(close_1h, EMA_PERIOD)
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    # Calculate 1-hour volume MA for confirmation
    volume_1h = df_1h['volume'].values
    volume_ma_1h = pd.Series(volume_1h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    
    # Calculate ATR for stoploss
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
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(ema_1h_aligned[i]) or np.isnan(volume_ma_1h_aligned[i]):
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
        
        # Volume confirmation: current 4h volume > 1.5x 1h volume MA
        # Note: 1h volume MA is aligned to 4h, so we compare 4h volume to 1h MA
        volume_ok = volume[i] > (volume_ma_1h_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1h_aligned[i]) else False
        
        # Trend filter: price above/below 1h EMA
        uptrend = close[i] > ema_1h_aligned[i]
        downtrend = close[i] < ema_1h_aligned[i]
        
        # Donchian breakout levels (using previous period's high/low to avoid look-ahead)
        # We use the highest high and lowest low of the past DONCHIAN_PERIOD periods
        # For 4h timeframe, we look back DONCHIAN_PERIOD candles
        if i >= DONCHIAN_PERIOD:
            donchian_high = np.max(high[i-DONCHIAN_PERIOD:i])
            donchian_low = np.min(low[i-DONCHIAN_PERIOD:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        # Breakout signals
        breakout_up = volume_ok and uptrend and (not np.isnan(donchian_high)) and (high[i] > donchian_high)
        breakout_down = volume_ok and downtrend and (not np.isnan(donchian_low)) and (low[i] < donchian_low)
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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

# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and 1-hour EMA trend filter.
# Donchian breakouts capture momentum bursts; volume confirms institutional participation;
# EMA ensures alignment with short-term trend to avoid whipsaws. Works in bull (breakouts above) and bear (breakdowns below).
# Target: 80-180 total trades over 4 years (20-45/year) to stay within optimal range.

name = "exp_13326_4h_donchian20_1h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1-hour data ONCE before loop for EMA and volume
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1-hour EMA for trend filter
    close_1h = df_1h['close'].values
    ema_1h = calculate_ema(close_1h, EMA_PERIOD)
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    # Calculate 1-hour volume MA for confirmation
    volume_1h = df_1h['volume'].values
    volume_ma_1h = pd.Series(volume_1h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    
    # Calculate ATR for stoploss
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
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(ema_1h_aligned[i]) or np.isnan(volume_ma_1h_aligned[i]):
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
        
        # Volume confirmation: current 4h volume > 1.5x 1h volume MA
        # Note: 1h volume MA is aligned to 4h, so we compare 4h volume to 1h MA
        volume_ok = volume[i] > (volume_ma_1h_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1h_aligned[i]) else False
        
        # Trend filter: price above/below 1h EMA
        uptrend = close[i] > ema_1h_aligned[i]
        downtrend = close[i] < ema_1h_aligned[i]
        
        # Donchian breakout levels (using previous period's high/low to avoid look-ahead)
        # We use the highest high and lowest low of the past DONCHIAN_PERIOD periods
        # For 4h timeframe, we look back DONCHIAN_PERIOD candles
        if i >= DONCHIAN_PERIOD:
            donchian_high = np.max(high[i-DONCHIAN_PERIOD:i])
            donchian_low = np.min(low[i-DONCHIAN_PERIOD:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        # Breakout signals
        breakout_up = volume_ok and uptrend and (not np.isnan(donchian_high)) and (high[i] > donchian_high)
        breakout_down = volume_ok and downtrend and (not np.isnan(donchian_low)) and (low[i] < donchian_low)
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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

# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and 1-hour EMA trend filter.
# Donchian breakouts capture momentum bursts; volume confirms institutional participation;
# EMA ensures alignment with short-term trend to avoid whipsaws. Works in bull (breakouts above) and bear (breakdowns below).
# Target: 80-180 total trades over 4 years (20-45/year) to stay within optimal range.

name = "exp_13326_4h_donchian20_1h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1-hour data ONCE before loop for EMA and volume
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1-hour EMA for trend filter
    close_1h = df_1h['close'].values
    ema_1h = calculate_ema(close_1h, EMA_PERIOD)
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    # Calculate 1-hour volume MA for confirmation
    volume_1h = df_1h['volume'].values
    volume_ma_1h = pd.Series(volume_1h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    
    # Calculate ATR for stoploss
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
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(ema_1h_aligned[i]) or np.isnan(volume_ma_1h_aligned[i]):
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
        
        # Volume confirmation: current 4h volume > 1.5x 1h volume MA
        # Note: 1h volume MA is aligned to 4h, so we compare 4h volume to 1h MA
        volume_ok = volume[i] > (volume_ma_1h_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1h_aligned[i]) else False
        
        # Trend filter: price above/below 1h EMA
        uptrend = close[i] > ema_1h_aligned[i]
        downtrend = close[i] < ema_1h_aligned[i]
        
        # Donchian breakout levels (using previous period's high/low to avoid look-ahead)
        # We use the highest high and lowest low of the past DONCHIAN_PERIOD periods
        # For 4h timeframe, we look back DONCHIAN_PERIOD candles
        if i >= DONCHIAN_PERIOD:
            donchian_high = np.max(high[i-DONCHIAN_PERIOD:i])
            donchian_low = np.min(low[i-DONCHIAN_PERIOD:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        # Breakout signals
        breakout_up = volume_ok and uptrend and (not np.isnan(donchian_high)) and (high[i] > donchian_high)
        breakout_down = volume_ok and downtrend and (not np.isnan(donchian_low)) and (low[i] < donchian_low)
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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

# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and 1-hour EMA trend filter.
# Donchian breakouts capture momentum bursts; volume confirms institutional participation;
# EMA ensures alignment with short-term trend to avoid whipsaws. Works in bull (breakouts above) and bear (breakdowns below).
# Target: 80-180 total trades over 4 years (20-45/year) to stay within optimal range.

name = "exp_13326_4h_donchian20_1h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1-hour data ONCE before loop for EMA and volume
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1-hour EMA for trend filter
    close_1h = df_1h['close'].values
    ema_1h = calculate_ema(close_1h, EMA_PERIOD)
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    # Calculate 1-hour volume MA for confirmation
    volume_1h = df_1h['volume'].values
    volume_ma_1h = pd.Series(volume_1h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    
    # Calculate ATR for stoploss
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
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(ema_1h_aligned[i]) or np.isnan(volume_ma_1h_aligned[i]):
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
        
        # Volume confirmation: current 4h volume > 1.5x 1h volume MA
        # Note: 1h volume MA is aligned to 4h, so we compare 4h volume to 1h MA
        volume_ok = volume[i] > (volume_ma_1h_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1h_aligned[i]) else False
        
        # Trend filter: price above/below 1h EMA
        uptrend = close[i] > ema_1h_aligned[i]
        downtrend = close[i] < ema_1h_aligned[i]
        
        # Donchian breakout levels (using previous period's high/low to avoid look-ahead)
        # We use the highest high and lowest low of the past DONCHIAN_PERIOD periods
        # For 4h timeframe, we look back DONCHIAN_PERIOD candles
        if i >= DONCHIAN_PERIOD:
            donchian_high = np.max(high[i-DONCHIAN_PERIOD:i])
            donchian_low = np.min(low[i-DONCHIAN_PERIOD:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        # Breakout signals
        breakout_up = volume_ok and uptrend and (not np.isnan(donchian_high)) and (high[i] > donchian_high)
        breakout_down = volume_ok and downtrend and (not np.isnan(donchian_low)) and (low[i] < donchian_low)
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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

------ EOF ------