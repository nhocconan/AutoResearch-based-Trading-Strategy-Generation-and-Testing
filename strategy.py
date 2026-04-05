#!/usr/bin/env python3
"""
Experiment #8315: 6-hour Williams %R with weekly pivot direction and volume confirmation.
Hypothesis: Williams %R identifies overbought/oversold conditions on 6h timeframe. 
When combined with weekly pivot direction (price above/below weekly pivot) and volume confirmation,
it captures mean reversion moves in ranging markets while avoiding false signals in trends.
Weekly pivot provides longer-term context to filter trades, reducing whipsaw in both bull and bear markets.
Targeting 75-200 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8315_6h_williamsr_weeklypivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
PIVOT_LOOKBACK = 5  # days to look back for weekly pivot
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.fillna(0).values

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: (H + L + C) / 3."""
    return (high + low + close) / 3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    price_above_pivot = close_1w > weekly_pivot  # True = bullish bias
    price_above_pivot_aligned = align_htf_to_ltf(prices, df_1w, price_above_pivot)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_R_PERIOD)
    
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
    target_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_R_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from weekly pivot
        bull_bias = price_above_pivot_aligned[i]  # price above weekly pivot
        bear_bias = ~price_above_pivot_aligned[i]  # price below weekly pivot
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Williams %R conditions
        oversold = williams_r[i] <= WILLIAMS_R_OVERSOLD
        overbought = williams_r[i] >= WILLIAMS_R_OVERBOUGHT
        
        # Entry conditions: mean reversion with bias
        long_entry = bull_bias and oversold and volume_confirmed
        short_entry = bear_bias and overbought and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
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
Experiment #8315: 6-hour Williams %R with weekly pivot direction and volume confirmation.
Hypothesis: Williams %R identifies overbought/oversold conditions on 6h timeframe. 
When combined with weekly pivot direction (price above/below weekly pivot) and volume confirmation,
it captures mean reversion moves in ranging markets while avoiding false signals in trends.
Weekly pivot provides longer-term context to filter trades, reducing whipsaw in both bull and bear markets.
Targeting 75-200 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8315_6h_williamsr_weeklypivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
PIVOT_LOOKBACK = 5  # days to look back for weekly pivot
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.fillna(0).values

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: (H + L + C) / 3."""
    return (high + low + close) / 3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    price_above_pivot = close_1w > weekly_pivot  # True = bullish bias
    price_above_pivot_aligned = align_htf_to_ltf(prices, df_1w, price_above_pivot)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_R_PERIOD)
    
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
    target_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_R_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from weekly pivot
        bull_bias = price_above_pivot_aligned[i]  # price above weekly pivot
        bear_bias = ~price_above_pivot_aligned[i]  # price below weekly pivot
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Williams %R conditions
        oversold = williams_r[i] <= WILLIAMS_R_OVERSOLD
        overbought = williams_r[i] >= WILLIAMS_R_OVERBOUGHT
        
        # Entry conditions: mean reversion with bias
        long_entry = bull_bias and oversold and volume_confirmed
        short_entry = bear_bias and overbought and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>