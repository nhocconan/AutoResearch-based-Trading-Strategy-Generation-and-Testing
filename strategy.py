#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray (Bull/Bear Power) + 12h Trend Filter
Hypothesis: Williams Alligator identifies trend presence and direction, Elder Ray measures bull/bear power strength.
Combined with 12h trend filter to avoid counter-trend trades. Works in bull markets (strong bull power above teeth)
and bear markets (strong bear power below teeth). Designed for 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6w_alligator_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_prev = np.roll(ema50_12h, 1)
    ema50_12h_prev[0] = ema50_12h[0]
    ema50_rising = ema50_12h > ema50_12h_prev
    ema50_falling = ema50_12h < ema50_12h_prev
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema50_falling)
    
    # Williams Alligator (13,8,5 SMAs with future shift)
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate SMAs with proper alignment (using typical price)
    typical_price = (high + low + close) / 3.0
    
    # Jaw (13)
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values
    # Teeth (8)
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values
    # Lips (5)
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values
    
    # Shift Alligator lines by bars to avoid look-ahead (Williams Alligator uses future values)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Fill NaN from rolling and shifting
    jaw[:13] = jaw[13] if len(jaw) > 13 else jaw[-1] if len(jaw) > 0 else 0
    teeth[:8] = teeth[8] if len(teeth) > 8 else teeth[-1] if len(teeth) > 0 else 0
    lips[:5] = lips[5] if len(lips) > 5 else lips[-1] if len(lips) > 0 else 0
    
    # Elder Ray (Bull Power/Bear Power)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For 12h EMA50 and Alligator
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Alligator sleep (jaws < teeth < lips for long, reverse for short) OR stoploss
        if position == 1:  # long position
            # Exit: Alligator sleeping (jaws below teeth below lips) OR stoploss
            if (jaw[i] < teeth[i] and teeth[i] < lips[i]) or \
               (close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator sleeping (jaws above teeth above lips) OR stoploss
            if (jaw[i] > teeth[i] and teeth[i] > lips[i]) or \
               (close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + Elder Ray + 12h trend
            # Alligator aligned for uptrend: lips > teeth > jaw
            # Alligator aligned for downtrend: jaws > teeth > lips
            alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Elder Ray: strong bull/bear power
            strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1]) * 1.5
            strong_bear = bear_power[i] < 0 and abs(bear_power[i]) > abs(np.mean(bear_power[max(0, i-20):i+1])) * 1.5
            
            # 12h trend filter
            uptrend_12h = ema50_rising_aligned[i]
            downtrend_12h = ema50_falling_aligned[i]
            
            if alligator_long and strong_bull and uptrend_12h:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif alligator_short and strong_bear and downtrend_12h:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray (Bull/Bear Power) + 12h Trend Filter
Hypothesis: Williams Alligator identifies trend presence and direction, Elder Ray measures bull/bear power strength.
Combined with 12h trend filter to avoid counter-trend trades. Works in bull markets (strong bull power above teeth)
and bear markets (strong bear power below teeth). Designed for 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6w_alligator_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_prev = np.roll(ema50_12h, 1)
    ema50_12h_prev[0] = ema50_12h[0]
    ema50_rising = ema50_12h > ema50_12h_prev
    ema50_falling = ema50_12h < ema50_12h_prev
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema50_falling)
    
    # Williams Alligator (13,8,5 SMAs with future shift)
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate SMAs with proper alignment (using typical price)
    typical_price = (high + low + close) / 3.0
    
    # Jaw (13)
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values
    # Teeth (8)
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values
    # Lips (5)
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values
    
    # Shift Alligator lines by bars to avoid look-ahead (Williams Alligator uses future values)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Fill NaN from rolling and shifting
    jaw[:13] = jaw[13] if len(jaw) > 13 else jaw[-1] if len(jaw) > 0 else 0
    teeth[:8] = teeth[8] if len(teeth) > 8 else teeth[-1] if len(teeth) > 0 else 0
    lips[:5] = lips[5] if len(lips) > 5 else lips[-1] if len(lips) > 0 else 0
    
    # Elder Ray (Bull Power/Bear Power)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For 12h EMA50 and Alligator
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Alligator sleep (jaws < teeth < lips for long, reverse for short) OR stoploss
        if position == 1:  # long position
            # Exit: Alligator sleeping (jaws below teeth below lips) OR stoploss
            if (jaw[i] < teeth[i] and teeth[i] < lips[i]) or \
               (close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator sleeping (jaws above teeth above lips) OR stoploss
            if (jaw[i] > teeth[i] and teeth[i] > lips[i]) or \
               (close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + Elder Ray + 12h trend
            # Alligator aligned for uptrend: lips > teeth > jaw
            # Alligator aligned for downtrend: jaws > teeth > lips
            alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Elder Ray: strong bull/bear power
            strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1]) * 1.5
            strong_bear = bear_power[i] < 0 and abs(bear_power[i]) > abs(np.mean(bear_power[max(0, i-20):i+1])) * 1.5
            
            # 12h trend filter
            uptrend_12h = ema50_rising_aligned[i]
            downtrend_12h = ema50_falling_aligned[i]
            
            if alligator_long and strong_bull and uptrend_12h:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif alligator_short and strong_bear and downtrend_12h:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals