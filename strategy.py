#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 12h trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) on 6h to detect trend changes, aligned with 12h EMA for trend filter.
# Volume confirmation ensures institutional participation. Works in trending markets (Alligator alignment)
# and avoids false signals in ranging markets. Target: 75-150 total trades over 4 years (19-38/year).

name = "exp_13479_6w_alligator_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_lines(close, period, jaw_shift, teeth_shift, lips_shift):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips"""
    # Smoothed median price (HL/2)
    median_price = (high + low) / 2
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=period, center=False).mean().shift(jaw_shift)
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=period-5, center=False).mean().shift(teeth_shift)
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=period-8, center=False).mean().shift(lips_shift)
    return jaw.values, teeth.values, lips.values

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator_lines(close, ALLIGATOR_PERIOD, JAW_SHIFT, TEETH_SHIFT, LIPS_SHIFT)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD + JAW_SHIFT, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_12h_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and alligator_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and alligator_short:
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

# Hypothesis: 6h Williams Alligator with 12h trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) on 6h to detect trend changes, aligned with 12h EMA for trend filter.
# Volume confirmation ensures institutional participation. Works in trending markets (Alligator alignment)
# and avoids false signals in ranging markets. Target: 75-150 total trades over 4 years (19-38/year).

name = "exp_13479_6w_alligator_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_lines(high, low, close, period, jaw_shift, teeth_shift, lips_shift):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips"""
    # Smoothed median price (HL/2)
    median_price = (high + low) / 2
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=period, center=False).mean().shift(jaw_shift)
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=period-5, center=False).mean().shift(teeth_shift)
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=period-8, center=False).mean().shift(lips_shift)
    return jaw.values, teeth.values, lips.values

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator_lines(high, low, close, ALLIGATOR_PERIOD, JAW_SHIFT, TEETH_SHIFT, LIPS_SHIFT)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD + JAW_SHIFT, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_12h_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and alligator_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and alligator_short:
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

# Hypothesis: 6h Williams Alligator with 12h trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) on 6h to detect trend changes, aligned with 12h EMA for trend filter.
# Volume confirmation ensures institutional participation. Works in trending markets (Alligator alignment)
# and avoids false signals in ranging markets. Target: 75-150 total trades over 4 years (19-38/year).

name = "exp_13479_6w_alligator_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_lines(high, low, close, period, jaw_shift, teeth_shift, lips_shift):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips"""
    # Smoothed median price (HL/2)
    median_price = (high + low) / 2
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=period, center=False).mean().shift(jaw_shift)
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=period-5, center=False).mean().shift(teeth_shift)
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=period-8, center=False).mean().shift(lips_shift)
    return jaw.values, teeth.values, lips.values

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator_lines(high, low, close, ALLIGATOR_PERIOD, JAW_SHIFT, TEETH_SHIFT, LIPS_SHIFT)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD + JAW_SHIFT, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_12h_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and alligator_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and alligator_short:
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

# Hypothesis: 6h Williams Alligator with 12h trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) on 6h to detect trend changes, aligned with 12h EMA for trend filter.
# Volume confirmation ensures institutional participation. Works in trending markets (Alligator alignment)
# and avoids false signals in ranging markets. Target: 75-150 total trades over 4 years (19-38/year).

name = "exp_13479_6w_alligator_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_lines(high, low, close, period, jaw_shift, teeth_shift, lips_shift):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips"""
    # Smoothed median price (HL/2)
    median_price = (high + low) / 2
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=period, center=False).mean().shift(jaw_shift)
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=period-5, center=False).mean().shift(teeth_shift)
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=period-8, center=False).mean().shift(lips_shift)
    return jaw.values, teeth.values, lips.values

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator_lines(high, low, close, ALLIGATOR_PERIOD, JAW_SHIFT, TEETH_SHIFT, LIPS_SHIFT)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD + JAW_SHIFT, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_12h_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and alligator_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and alligator_short:
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

---  END OF FILE  ---