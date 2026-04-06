#!/usr/bin/env python3
"""
Experiment #11771: 6h Williams Alligator + Elder Ray + 1d Trend Filter
Hypothesis: Williams Alligator identifies trending vs ranging markets on 6h, Elder Ray measures bull/bear power,
and 1d EMA provides higher timeframe trend filter. This combination reduces false signals in ranging markets
while capturing trends, working in both bull (trend following) and bear (counter-trend reversals) by adapting
to market regime via Alligator's jaw/teeth/lips alignment. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11771_6h_alligator_elder_1dtrend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(close, period_jaw, period_teeth, period_lips):
    """Calculate Williams Alligator: jaw (13), teeth (8), lips (5) SMAs with future shift"""
    jaw = pd.Series(close).rolling(window=period_jaw, center=False).mean().shift(period_jaw//2).values
    teeth = pd.Series(close).rolling(window=period_teeth, center=False).mean().shift(period_teeth//2).values
    lips = pd.Series(close).rolling(window=period_lips, center=False).mean().shift(period_lips//2).values
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

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
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: max of all periods + shifts
    start = max(ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS, 
                ELDER_RAY_PERIOD, TREND_EMA_PERIOD) + ALLIGATOR_PERIOD_JAW//2 + 1
    
    for i in range(start, n):
        # Skip if 1d EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Alligator signals: lips above teeth above jaw = uptrend, reverse for downtrend
        alligator_long = lips[i] > teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray: bull power > 0 and bear power < 0 for strong trend
        elder_long = bull_power[i] > 0 and bear_power[i] < 0
        elder_short = bull_power[i] < 0 and bear_power[i] > 0
        
        # Trend filter (1d)
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions - require alignment
        long_entry = alligator_long and elder_long and uptrend_1d
        short_entry = alligator_short and elder_short and downtrend_1d
        
        # Exit conditions - any disagreement
        long_exit = not (alligator_long and elder_long)
        short_exit = not (alligator_short and elder_short)
        
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
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>

#!/usr/bin/env python3
"""
Experiment #11771: 6h Williams Alligator + Elder Ray + 1d Trend Filter
Hypothesis: Williams Alligator identifies trending vs ranging markets on 6h, Elder Ray measures bull/bear power,
and 1d EMA provides higher timeframe trend filter. This combination reduces false signals in ranging markets
while capturing trends, working in both bull (trend following) and bear (counter-trend reversals) by adapting
to market regime via Alligator's jaw/teeth/lips alignment. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11771_6h_alligator_elder_1dtrend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(close, period_jaw, period_teeth, period_lips):
    """Calculate Williams Alligator: jaw (13), teeth (8), lips (5) SMAs with future shift"""
    jaw = pd.Series(close).rolling(window=period_jaw, center=False).mean().shift(period_jaw//2).values
    teeth = pd.Series(close).rolling(window=period_teeth, center=False).mean().shift(period_teeth//2).values
    lips = pd.Series(close).rolling(window=period_lips, center=False).mean().shift(period_lips//2).values
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

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
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: max of all periods + shifts
    start = max(ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS, 
                ELDER_RAY_PERIOD, TREND_EMA_PERIOD) + ALLIGATOR_PERIOD_JAW//2 + 1
    
    for i in range(start, n):
        # Skip if 1d EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Alligator signals: lips above teeth above jaw = uptrend, reverse for downtrend
        alligator_long = lips[i] > teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray: bull power > 0 and bear power < 0 for strong trend
        elder_long = bull_power[i] > 0 and bear_power[i] < 0
        elder_short = bull_power[i] < 0 and bear_power[i] > 0
        
        # Trend filter (1d)
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions - require alignment
        long_entry = alligator_long and elder_long and uptrend_1d
        short_entry = alligator_short and elder_short and downtrend_1d
        
        # Exit conditions - any disagreement
        long_exit = not (alligator_long and elder_long)
        short_exit = not (alligator_short and elder_short)
        
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
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals