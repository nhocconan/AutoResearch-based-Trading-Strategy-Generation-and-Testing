#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Bollinger Band squeeze breakout with 1-week trend filter.
# Bollinger Band squeeze (low volatility) precedes breakouts in both bull and bear markets.
# The weekly EMA ensures alignment with higher timeframe momentum.
# Volume confirmation filters false breakouts.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "exp_13218_1d_bb_squeeze_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
SQUEEZE_THRESHOLD = 0.03  # Bandwidth threshold for squeeze
EMA_PERIOD = 50  # Weekly EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_bollinger_bands(close, period, std):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std_dev = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    upper, lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    bandwidth = (upper - lower) / pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Squeeze condition: bandwidth below threshold
        squeeze = bandwidth[i] < SQUEEZE_THRESHOLD if not np.isnan(bandwidth[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals: price breaks Bollinger Band after squeeze
        breakout_up = squeeze and volume_ok and uptrend and (close[i] > upper[i-1])
        breakout_down = squeeze and volume_ok and downtrend and (close[i] < lower[i-1])
        
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

# Hypothesis: 1-day KAMA trend with 1-week EMA filter and volume confirmation.
# KAMA adapts to market noise, reducing false signals in sideways markets.
# Weekly EMA ensures alignment with higher timeframe momentum.
# Volume confirms genuine breakouts.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "exp_13218_1d_kama_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_ER_PERIOD = 10
KAMA_FAST = 2
KAMA_SLOW = 30
EMA_PERIOD = 50  # Weekly EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_kama(close, er_period, fast, slow):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[er_period:] = change[er_period-1:] / volatility[er_period-1:]
    sc = np.square(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_ema(close, period):
    """Calculate EMA"""
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA
    kama = calculate_kama(close, KAMA_ER_PERIOD, KAMA_FAST, KAMA_SLOW)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_ER_PERIOD + 1, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # KAMA slope: rising/falling
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Entry signals: KAMA direction with volume and trend
        entry_long = kama_rising and volume_ok and uptrend
        entry_short = kama_falling and volume_ok and downtrend
        
        # Generate signals
        if position == 0:
            if entry_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif entry_short:
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

# Hypothesis: 1-day ADX trend strength with 1-week EMA filter and volume confirmation.
# ADX > 25 indicates strong trend, effective in both bull and bear markets.
# Weekly EMA ensures alignment with higher timeframe momentum.
# Volume confirms genuine trend strength.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "exp_13218_1d_adx_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
EMA_PERIOD = 50  # Weekly EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # Directional Index
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_ema(close, period):
    """Calculate EMA"""
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # ADX trend strength
        trending = adx[i] > ADX_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Entry signals: ADX trend with volume and price alignment
        entry_long = trending and volume_ok and uptrend
        entry_short = trending and volume_ok and downtrend
        
        # Generate signals
        if position == 0:
            if entry_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif entry_short:
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

# Hypothesis: 1-day Williams %R mean reversion with 1-week EMA filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; mean reversion works in ranging markets.
# Weekly EMA ensures alignment with higher timeframe momentum.
# Volume confirms genuine reversals.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "exp_13218_1d_williamsr_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
OVERBOUGHT = -20
OVERSOLD = -80
EMA_PERIOD = 50  # Weekly EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r

def calculate_ema(close, period):
    """Calculate EMA"""
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        oversold = williams_r[i] < OVERSOLD
        overbought = williams_r[i] > OVERBOUGHT
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Entry signals: Williams %R reversal with volume and trend alignment
        # In uptrend, buy oversold; in downtrend, sell overbought
        entry_long = oversold and volume_ok and uptrend
        entry_short = overbought and volume_ok and downtrend
        
        # Generate signals
        if position == 0:
            if entry_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif entry_short:
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

# Hypothesis: 1-day Aroon oscillator trend with 1-week EMA filter and volume confirmation.
# Aroon identifies trend strength and potential reversals.
# Weekly EMA ensures alignment with higher timeframe momentum.
# Volume confirms genuine trend strength.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "exp_13218_1d_aroon_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
AROON_PERIOD = 25
AROON_THRESHOLD = 50  # Above 50 indicates strong trend
EMA_PERIOD = 50  # Weekly EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_aroon(high, low, period):
    """Calculate Aroon Up and Down"""
    # Aroon Up: measures strength of uptrend
    aroon_up = np.zeros_like(high, dtype=float)
    # Aroon Down: measures strength of downtrend
    aroon_down = np.zeros_like(low, dtype=float)
    
    for i in range(period, len(high)):
        # Period high
        period_high = np.max(high[i-period+1:i+1])
        # Period low
        period_low = np.min(low[i-period+1:i+1])
        
        # Days since period high
        days_since_high = np.where(high[i-period+1:i+1] == period_high)[0]
        # Days since period low
        days_since_low = np.where(low[i-period+1:i+1] == period_low)[0]
        
        if len(days_since_high) > 0:
            aroon_up[i] = ((period - 1 - days_since_high[-1]) / (period - 1)) * 100
        if len(days_since_low) > 0:
            aroon_down[i] = ((period - 1 - days_since_low[-1]) / (period - 1)) * 100
    
    return aroon_up, aroon_down

def calculate_ema(close, period):
    """Calculate EMA"""
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Aroon
    aroon_up, aroon_down = calculate_aroon(high, low, AROON_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(AROON_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Aroon trend strength: Aroon Up > Down and both above threshold for uptrend
        # Aroon Down > Up and both above threshold for downtrend
        strong_uptrend = (aroon_up[i] > aroon_down[i]) and (aroon_up[i] > AROON_THRESHOLD)
        strong_downtrend = (aroon_down[i] > aroon_up[i]) and (aroon_down[i] > AROON_THRESHOLD)
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Entry signals: Aroon trend with volume and price alignment
        entry_long = strong_uptrend and volume_ok and uptrend
        entry_short = strong_downtrend and volume_ok and downtrend
        
        # Generate signals
        if position == 0:
            if entry_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif entry_short:
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

--- END OF FILE ---