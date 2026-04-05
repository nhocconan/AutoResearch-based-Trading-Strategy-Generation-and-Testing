#!/usr/bin/env python3
"""
Experiment #7919: 6-hour Williams Alligator with 12h Elder Ray and volume confirmation.
Hypothesis: The Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (sleeping) or awakening on 6h.
Elder Ray (Bull/Bear Power) on 12h provides trend strength/direction. Volume confirms participation.
Enter long when: 6h Lips > Teeth > Jaw (bullish alignment) AND 12h Bull Power > 0 AND volume > 1.5x 20MA.
Enter short when: 6s Lips < Teeth < Jaw (bearish alignment) AND 12h Bear Power < 0 AND volume > 1.5x 20MA.
Exit on opposite signal or ATR(14) stop (2x). Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7919_6h_williams_alligator_12h_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13  # Williams Alligator default
ELDER_RAY_PERIOD = 13  # EMA period for Elder Ray
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(close, period):
    """Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMMA"""
    # Smoothed Moving Average (SMMA) approximation using EMA
    jaw = pd.Series(close).ewm(span=period*2, adjust=False, min_periods=period*2).mean().values
    teeth = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    lips = pd.Series(close).ewm(span=period//2, adjust=False, min_periods=period//2).mean().values
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, ema_period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Elder Ray)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Elder Ray components
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    bull_power_12h, bear_power_12h, ema_12h = calculate_elder_ray(high_12h, low_12h, close_12h, ELDER_RAY_PERIOD)
    elder_ray_signal = np.where(bull_power_12h > 0, 1, np.where(bear_power_12h < 0, -1, 0))  # 1=bull, -1=bear, 0=neutral
    elder_ray_aligned = align_htf_to_ltf(prices, df_12h, elder_ray_signal)
    
    # Calculate 6h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD)
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD*2, ELDER_RAY_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(elder_ray_aligned[i]):
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
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bullish_alignment[i] and (elder_ray_aligned[i] == 1) and volume_confirmed
        short_entry = bearish_alignment[i] and (elder_ray_aligned[i] == -1) and volume_confirmed
        
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
Experiment #7919: 6-hour Williams Alligator with 12h Elder Ray and volume confirmation.
Hypothesis: The Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (sleeping) or awakening on 6h.
Elder Ray (Bull/Bear Power) on 12h provides trend strength/direction. Volume confirms participation.
Enter long when: 6h Lips > Teeth > Jaw (bullish alignment) AND 12h Bull Power > 0 AND volume > 1.5x 20MA.
Enter short when: 6h Lips < Teeth < Jaw (bearish alignment) AND 12h Bear Power < 0 AND volume > 1.5x 20MA.
Exit on opposite signal or ATR(14) stop (2x). Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7919_6h_williams_alligator_12h_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13  # Williams Alligator default
ELDER_RAY_PERIOD = 13  # EMA period for Elder Ray
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(close, period):
    """Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMMA"""
    # Smoothed Moving Average (SMMA) approximation using EMA
    jaw = pd.Series(close).ewm(span=period*2, adjust=False, min_periods=period*2).mean().values
    teeth = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    lips = pd.Series(close).ewm(span=period//2, adjust=False, min_periods=period//2).mean().values
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, ema_period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Elder Ray)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Elder Ray components
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    bull_power_12h, bear_power_12h, ema_12h = calculate_elder_ray(high_12h, low_12h, close_12h, ELDER_RAY_PERIOD)
    elder_ray_signal = np.where(bull_power_12h > 0, 1, np.where(bear_power_12h < 0, -1, 0))  # 1=bull, -1=bear, 0=neutral
    elder_ray_aligned = align_htf_to_ltf(prices, df_12h, elder_ray_signal)
    
    # Calculate 6h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD)
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD*2, ELDER_RAY_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(elder_ray_aligned[i]):
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
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bullish_alignment[i] and (elder_ray_aligned[i] == 1) and volume_confirmed
        short_entry = bearish_alignment[i] and (elder_ray_aligned[i] == -1) and volume_confirmed
        
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