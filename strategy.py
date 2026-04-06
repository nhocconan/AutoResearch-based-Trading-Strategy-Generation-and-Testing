#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1w trend filter
# Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND 1w EMA(21) rising
# Short when: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND 1w EMA(21) falling
# Exit when: Alligator lines cross in opposite direction OR power signals reverse
# Uses weekly trend to filter trades, targeting 80-150 trades over 4 years

name = "6h_alligator_elder_1wtrend_v1"
timeframe = "6h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator on 6h (periods: Jaw=13, Teeth=8, Lips=5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 1w EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Trend direction: rising/falling EMA
    ema_rising = np.full(n, False)
    ema_falling = np.full(n, False)
    for i in range(1, n):
        if not np.isnan(ema_21_1w_aligned[i]) and not np.isnan(ema_21_1w_aligned[i-1]):
            ema_rising[i] = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
            ema_falling[i] = ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bearish Alligator alignment OR Bear Power negative
            if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bullish Alligator alignment OR Bull Power positive
            if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + Power signals + Weekly trend
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            if bullish_align and bull_power[i] > 0 and ema_rising[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_align and bear_power[i] < 0 and ema_falling[i]:
                signals[i] = -0.25
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1w trend filter
# Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND 1w EMA(21) rising
# Short when: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND 1w EMA(21) falling
# Exit when: Alligator lines cross in opposite direction OR power signals reverse
# Uses weekly trend to filter trades, targeting 80-150 trades over 4 years

name = "6h_alligator_elder_1wtrend_v1"
timeframe = "6h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator on 6h (periods: Jaw=13, Teeth=8, Lips=5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 1w EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Trend direction: rising/falling EMA
    ema_rising = np.full(n, False)
    ema_falling = np.full(n, False)
    for i in range(1, n):
        if not np.isnan(ema_21_1w_aligned[i]) and not np.isnan(ema_21_1w_aligned[i-1]):
            ema_rising[i] = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
            ema_falling[i] = ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bearish Alligator alignment OR Bear Power negative
            if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bullish Alligator alignment OR Bull Power positive
            if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + Power signals + Weekly trend
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            if bullish_align and bull_power[i] > 0 and ema_rising[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_align and bear_power[i] < 0 and ema_falling[i]:
                signals[i] = -0.25
                position = -1
    
    return signals