#!/usr/bin/env python3
"""
Experiment #8159: 6-hour Williams Alligator + Elder Ray with 12h trend filter
Hypothesis: Williams Alligator identifies trend direction and strength, while Elder Ray measures bull/bear power.
Combining with 12h EMA trend filter filters out counter-trend signals. Works in both bull/bear by only taking
trades in direction of higher timeframe trend. Williams Alligator's smoothed averages reduce whipsaw.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8159_6w_alligator_elder_12h_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed SMA (blue line)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed SMA (red line)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed SMA (green line)
ELDER_RAY_PERIOD = 13       # EMA for Elder Ray calculation
EMA_TREND_PERIOD = 50       # 12h EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smoothed_ma(series, period):
    """Williams Alligator uses SMMA (Smoothed Moving Average)"""
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(series, np.nan, dtype=float)
    if len(series) >= period:
        smma[period-1] = sma[period-1]
        for i in range(period, len(series)):
            smma[i] = (smma[i-1] * (period-1) + series[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    price_above_ema = close_12h > ema_12h  # True for uptrend bias
    price_above_ema_aligned = align_htf_to_ltf(prices, df_12h, price_above_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator (SMMA)
    jaw = calculate_smoothed_ma(close, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_smoothed_ma(close, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_smoothed_ma(close, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = EMA - Low
    ema_elder = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema_elder
    bear_power = ema_elder - low
    
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
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD,
                ELDER_RAY_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_ema_aligned[i]):
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
        
        # Alligator alignment: check if lines are properly ordered
        # In uptrend: Lips > Teeth > Jaw
        # In downtrend: Jaw > Teeth > Lips
        alligator_long = (not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(jaw[i]) and
                         lips[i] > teeth[i] and teeth[i] > jaw[i])
        alligator_short = (not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(jaw[i]) and
                          jaw[i] > teeth[i] and teeth[i] > lips[i])
        
        # Elder Ray confirmation
        elder_long = bull_power[i] > 0  # Bull power positive
        elder_short = bear_power[i] > 0  # Bear power positive
        
        # Trend filter from 12h
        uptrend_filter = price_above_ema_aligned[i]
        downtrend_filter = ~price_above_ema_aligned[i]
        
        # Entry conditions: Alligator direction + Elder Ray + 12h trend
        long_entry = alligator_long and elder_long and uptrend_filter
        short_entry = alligator_short and elder_short and downtrend_filter
        
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
Experiment #8159: 6-hour Williams Alligator + Elder Ray with 12h trend filter
Hypothesis: Williams Alligator identifies trend direction and strength, while Elder Ray measures bull/bear power.
Combining with 12h EMA trend filter filters out counter-trend signals. Works in both bull/bear by only taking
trades in direction of higher timeframe trend. Williams Alligator's smoothed averages reduce whipsaw.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8159_6w_alligator_elder_12h_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed SMA (blue line)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed SMA (red line)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed SMA (green line)
ELDER_RAY_PERIOD = 13       # EMA for Elder Ray calculation
EMA_TREND_PERIOD = 50       # 12h EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smoothed_ma(series, period):
    """Williams Alligator uses SMMA (Smoothed Moving Average)"""
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(series, np.nan, dtype=float)
    if len(series) >= period:
        smma[period-1] = sma[period-1]
        for i in range(period, len(series)):
            smma[i] = (smma[i-1] * (period-1) + series[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    price_above_ema = close_12h > ema_12h  # True for uptrend bias
    price_above_ema_aligned = align_htf_to_ltf(prices, df_12h, price_above_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator (SMMA)
    jaw = calculate_smoothed_ma(close, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_smoothed_ma(close, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_smoothed_ma(close, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = EMA - Low
    ema_elder = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema_elder
    bear_power = ema_elder - low
    
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
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD,
                ELDER_RAY_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_ema_aligned[i]):
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
        
        # Alligator alignment: check if lines are properly ordered
        # In uptrend: Lips > Teeth > Jaw
        # In downtrend: Jaw > Teeth > Lips
        alligator_long = (not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(jaw[i]) and
                         lips[i] > teeth[i] and teeth[i] > jaw[i])
        alligator_short = (not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(jaw[i]) and
                          jaw[i] > teeth[i] and teeth[i] > lips[i])
        
        # Elder Ray confirmation
        elder_long = bull_power[i] > 0  # Bull power positive
        elder_short = bear_power[i] > 0  # Bear power positive
        
        # Trend filter from 12h
        uptrend_filter = price_above_ema_aligned[i]
        downtrend_filter = ~price_above_ema_aligned[i]
        
        # Entry conditions: Alligator direction + Elder Ray + 12h trend
        long_entry = alligator_long and elder_long and uptrend_filter
        short_entry = alligator_short and elder_short and downtrend_filter
        
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