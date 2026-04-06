#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h ADX trend filter (ADX > 25) with 6h Williams Alligator crossover for entry.
# Goes long when Alligator Jaw (13-period SMMA) crosses above Teeth (8-period SMMA) and Lips (5-period SMMA) with ADX > 25,
# short when Jaw crosses below Teeth and Lips with ADX > 25.
# Uses ATR-based stop loss to manage risk. Designed for 50-150 total trades over 4 years (12-37/year).
# Williams Alligator provides clear trend-following signals, ADX filters for trending conditions only,
# reducing whipsaws in sideways markets.

name = "exp_13859_6h_williams_alligator_adx_12h_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SMMA_PERIOD = 2  # Smoothing period for SMMA
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    return pd.Series(data).ewm(alpha=1/period, adjust=False).mean().values

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index (ADX)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
    
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for ADX trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, ADX_PERIOD)
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h data for Williams Alligator and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator lines (SMMA)
    jaw = calculate_smma(high, ALLIGATOR_JAW_PERIOD)  # Typically uses median price
    teeth = calculate_smma(low, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_smma(close, ALLIGATOR_LIPS_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Trend filter: ADX > threshold
        trending = adx_12h_aligned[i] > ADX_THRESHOLD
        
        # Williams Alligator crossover signals
        jaw_above_teeth = jaw[i] > teeth[i]
        teeth_above_lips = teeth[i] > lips[i]
        jaw_below_teeth = jaw[i] < teeth[i]
        teeth_below_lips = teeth[i] < lips[i]
        
        # Previous values for crossover detection
        jaw_above_teeth_prev = jaw[i-1] > teeth[i-1]
        teeth_above_lips_prev = teeth[i-1] > lips[i-1]
        jaw_below_teeth_prev = jaw[i-1] < teeth[i-1]
        teeth_below_lips_prev = teeth[i-1] < lips[i-1]
        
        # Bullish crossover: Jaw crosses above Teeth and Teeth above Lips
        bullish_cross = (jaw_above_teeth and teeth_above_lips) and (not jaw_above_teeth_prev or not teeth_above_lips_prev)
        # Bearish crossover: Jaw crosses below Teeth and Teeth below Lips
        bearish_cross = (jaw_below_teeth and teeth_below_lips) and (not jaw_below_teeth_prev or not teeth_below_lips_prev)
        
        # Generate signals
        if position == 0:
            if trending and bullish_cross:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif trending and bearish_cross:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish crossover
            if bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on bullish crossover
            if bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
# Goes long when price breaks above 12h Donchian upper band with above-average volume and price above 1d EMA200,
# short when breaks below 12h Donchian lower band with volume and price below 1d EMA200.
# Uses ATR-based stop loss to manage risk. Designed for 50-150 total trades over 4 years (12-37/year).
# Donchian channels provide clear structure, EMA200 filters trend direction, volume confirms breakout strength.

name = "exp_13859_6h_donchian20_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
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
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    
    # Align 1d EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data for Donchian channels, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels on 6h data
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Donchian breakout signals
        long_signal = volume_ok and above_ema and close[i] > upper[i]
        short_signal = volume_ok and below_ema and close[i] < lower[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below Donchian lower band
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above Donchian upper band
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4.
# Calculates Camarilla levels from prior 1d OHLC: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
# Goes long when price crosses above R4 with volume confirmation, short when crosses below S4.
# Fades at R3/S3: short at R3 with volume, long at S3 with volume.
# Uses ATR-based stop loss. Designed for 50-150 total trades over 4 years (12-37/year).
# Camarilla pivots provide institutional support/resistance levels with clear fade/breakout rules.

name = "exp_13859_6h_camarilla1d_vol_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels from prior period OHLC"""
    # Typical Camarilla formula based on previous day's range
    range_ = high - low
    c = close
    r4 = c + (range_ * CAMARILLA_MULTIPLIER / 2)
    r3 = c + (range_ * CAMARILLA_MULTIPLIER / 4)
    s3 = c - (range_ * CAMARILLA_MULTIPLIER / 4)
    s4 = c - (range_ * CAMARILLA_MULTIPLIER / 2)
    return r4, r3, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for price, volume, and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need at least 1 day of data)
    start = 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Camarilla signal logic
        # Breakout continuation: long at R4 break, short at S4 break
        long_breakout = volume_ok and close[i] > r4_1d_aligned[i] and close[i-1] <= r4_1d_aligned[i-1]
        short_breakout = volume_ok and close[i] < s4_1d_aligned[i] and close[i-1] >= s4_1d_aligned[i-1]
        
        # Fade at R3/S3: short at R3 resistance, long at S3 support
        short_fade = volume_ok and close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1]
        long_fade = volume_ok and close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1]
        
        # Generate signals
        if position == 0:
            if long_breakout or long_fade:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakout or short_fade:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below S3 (support broken) or at R3 (take profit)
            if close[i] < s3_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above R3 (resistance broken) or at S3 (take profit)
            if close[i] > r3_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend (ATR=10, mult=3) for trend direction and 6h EMA(8/21) crossover for entry timing.
# Goes long when 12h Supertrend is bullish and 6h EMA8 crosses above EMA21, short when 12h Supertrend is bearish and EMA8 crosses below EMA21.
# Uses ATR-based stop loss to manage risk. Designed for 50-150 total trades over 4 years (12-37/year).
# Supertrend provides reliable trend filtration, EMA crossover provides timely entries in trending markets.

name = "exp_13859_6h_supertrend12h_ema8_21_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3
EMA_FAST = 8
EMA_SLOW = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_supertrend(high, low, close, atr_period, multiplier):
    """Calculate Supertrend indicator"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).mean().values
    
    hl2 = (high + low) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upper[i-1]:
            direction[i] = 1
        elif close[i] < lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower[i] < lower[i-1]:
                lower[i] = lower[i-1]
            if direction[i] == -1 and upper[i] > upper[i-1]:
                upper[i] = upper[i-1]
        
        supertrend[i] = upper[i] if direction[i] == 1 else lower[i]
    
    return supertrend, direction

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Supertrend trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Supertrend for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    supertrend_12h, direction_12h = calculate_supertrend(high_12h, low_12h, close_12h, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER)
    
    # Align 12h Supertrend direction to 6h timeframe
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # 6h data for EMA crossover and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA crossover
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SUPERTREND_ATR_PERIOD, EMA_FAST, EMA_SLOW) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(direction_12h_aligned[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Trend filter: 12h Supertrend direction
        bullish_trend = direction_12h_aligned[i] == 1
        bearish_trend = direction_12h_aligned[i] == -1
        
        # EMA crossover signals
        ema_fast_above = ema_fast[i] > ema_slow[i]
        ema_fast_below = ema_fast[i] < ema_slow[i]
        
        # Previous values for crossover detection
        ema_fast_above_prev = ema_fast[i-1] > ema_slow[i-1]
        ema_fast_below_prev = ema_fast[i-1] < ema_slow[i-1]
        
        # Bullish crossover: fast crosses above slow
        bullish_cross = ema_fast_above and not ema_fast_above_prev
        # Bearish crossover: fast crosses below slow
        bearish_cross = ema_fast_below and not ema_fast_below_prev
        
        # Generate signals
        if position == 0:
            if bullish_trend and bullish_cross:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_trend and bearish_cross:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish crossover or trend change
            if bearish_cross or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on bullish crossover or trend change
            if bullish_cross or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h KAMA (adaptive EMA) with 6h Bollinger Bands squeeze and volume breakout.
# Goes long when 12h KAMA is trending upward, 6h Bollinger Bands width is below 20th percentile (squeeze), and price breaks above upper band with volume.
# Short when 12h KAMA is trending downward, BB squeeze, and price breaks below lower band with volume.
# Uses ATR-based stop loss. Designed for 50-150 total trades over 4 years (12-37/year).
# KAMA adapts to market efficiency, Bollinger Squeeze identifies low volatility breakout setups.

name = "exp_13859_6h_kama12h_bb_squeeze_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
KAMA_PERIOD = 10
KAMA_FAST = 2
KAMA_SLOW = 30
BB_PERIOD = 20
BB_STD_DEV = 2
BB_SQUEEZE_PERCENTILE = 20  # Below 20th percentile indicates squeeze
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_er(close, period):
    """Calculate Efficiency Ratio for KAMA"""
    change = np.abs(close - np.roll(close, period))
    change[0:period] = 0
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Proper ER calculation: |close - close[period]| / sum(|diff|) over period
    er = np.zeros_like(close)
    for i in range(period, len(close)):
        price_change = np.abs(close[i] - close[i-period])
        volatility_sum = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility_sum > 0:
            er[i] = price_change / volatility_sum
        else:
            er[i] = 0
    return er

def calculate_kama(close, period, fast, slow):
    """Calculate Kaufman Adaptive Moving Average"""
    er = calculate_er(close