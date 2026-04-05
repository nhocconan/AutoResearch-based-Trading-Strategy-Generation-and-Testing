#!/usr/bin/env python3
"""
Experiment #8515: 6h Williams Alligator + Elder Ray + ADX Regime
Hypothesis: The Williams Alligator (jaw/teeth/lips) identifies trend direction and strength, 
while Elder Ray (bull/bear power) measures trend momentum. Combined with ADX regime filtering,
this creates a robust trend-following system that works in both bull and bear markets by 
only trading when the Alligator is 'awake' (trending) and ADX confirms strong trend.
The Elder Ray filter ensures we only take trades in the direction of underlying power,
reducing false signals during weak trends. Targets 50-150 total trades over 4 years (12-37/year)
to minimize fee drag while maintaining statistical validity.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8515_6h_alligator_elder_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13  # Smoothed with 8-period shift
ALLIGATOR_PERIOD_TEETH = 8  # Smoothed with 5-period shift
ALLIGATOR_PERIOD_LIPS = 5   # Smoothed with 3-period shift
ELDER_RAY_PERIOD = 13       # EMA for Bull/Bear Power calculation
ADX_PERIOD = 14             # Standard ADX period
ADX_THRESHOLD = 25          # Minimum ADX for trending market
EMA_FAST = 12               # For EMA calculation in Elder Ray
EMA_SLOW = 26               # For EMA calculation in Elder Ray
SIGNAL_SIZE = 0.25          # Position size (25% of capital)

def calculate_wma(series, period):
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return np.convolve(series, weights / weights.sum(), mode='same')

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines (Jaw, Teeth, Lips)"""
    # Median price
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).ewm(alpha=1/ALLIGATOR_PERIOD_JAW, adjust=False, min_periods=ALLIGATOR_PERIOD_JAW).mean().values
    jaw = np.roll(jaw, -ALLIGATOR_PERIOD_JAW//2)  # Center the smoothed line
    
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).ewm(alpha=1/ALLIGATOR_PERIOD_TEETH, adjust=False, min_periods=ALLIGATOR_PERIOD_TEETH).mean().values
    teeth = np.roll(teeth, -ALLIGATOR_PERIOD_TEETH//2)
    
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).ewm(alpha=1/ALLIGATOR_PERIOD_LIPS, adjust=False, min_periods=ALLIGATOR_PERIOD_LIPS).mean().values
    lips = np.roll(lips, -ALLIGATOR_PERIOD_LIPS//2)
    
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close):
    """Calculate Elder Ray: Bull Power and Bear Power"""
    # Calculate 13-period EMA
    ema = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    
    # Bull Power = High - EMA
    bull_power = high - ema
    
    # Bear Power = Low - EMA
    bear_power = low - ema
    
    return bull_power, bear_power

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_period
    di_minus = 100 * dm_minus_smooth / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams Alligator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high_1d, low_1d, close_1d)
    
    # Alligator is 'awake' when lips > teeth > jaw (uptrend) or lips < teeth < jaw (downtrend)
    alligator_up = (lips_1d > teeth_1d) & (teeth_1d > jaw_1d)
    alligator_down = (lips_1d < teeth_1d) & (teeth_1d < jaw_1d)
    alligator_awake = alligator_up | alligator_down
    
    # Determine trend direction from Alligator
    alligator_direction = np.where(alligator_up, 1, np.where(alligator_down, -1, 0))
    alligator_direction_aligned = align_htf_to_ltf(prices, df_1d, alligator_direction)
    
    # Calculate 1d Elder Ray
    bull_power_1d, bear_power_1d = calculate_elder_ray(high_1d, low_1d, close_1d)
    
    # Elder Ray signals: Bull Power > 0 and rising, Bear Power < 0 and falling
    bull_power_rising = bull_power_1d > np.roll(bull_power_1d, 1)
    bear_power_falling = bear_power_1d < np.roll(bear_power_1d, 1)
    elder_bull = bull_power_1d > 0 & bull_power_rising
    elder_bear = bear_power_1d < 0 & bear_power_falling
    
    # Elder Ray direction
    elder_direction = np.where(elder_bull, 1, np.where(elder_bear, -1, 0))
    elder_direction_aligned = align_htf_to_ltf(prices, df_1d, elder_direction)
    
    # Calculate 1d ADX for regime filtering
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    strong_trend = adx_1d >= ADX_THRESHOLD
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator on 6h for entry timing
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Entry signals based on Alligator crossover
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # Bullish crossover: lips crosses above teeth
    bullish_crossover = lips_above_teeth & ~np.roll(lips_above_teeth, 1)
    # Bearish crossover: lips crosses below teeth
    bearish_crossover = lips_below_teeth & ~np.roll(lips_below_teeth, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS,
                ELDER_RAY_PERIOD, ADX_PERIOD) + 10
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(alligator_direction_aligned[i]) or np.isnan(elder_direction_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Check for trend reversal (Alligator sleeping)
        if alligator_direction_aligned[i] == 0:  # Alligator sleeping
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trade direction based on HTF alignment
        htf_alligator_dir = alligator_direction_aligned[i]
        htf_elder_dir = elder_direction_aligned[i]
        htf_strong_trend = strong_trend_aligned[i]
        
        # Require alignment between Alligator direction and Elder Ray
        direction_aligned = (htf_alligator_dir == htf_elder_dir) and (htf_alligator_dir != 0)
        
        # Only trade in strong trend regimes
        if not (direction_aligned and htf_strong_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check for entry signals on 6h timeframe
        if position == 0:
            # Long entry: bullish crossover in alignment with bullish HTF
            if htf_alligator_dir == 1 and bullish_crossover[i]:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            # Short entry: bearish crossover in alignment with bearish HTF
            elif htf_alligator_dir == -1 and bearish_crossover[i]:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Maintain long position
            signals[i] = SIGNAL_SIZE
            # Exit if Alligator goes to sleep or reverses
            if alligator_direction_aligned[i] != 1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Maintain short position
            signals[i] = -SIGNAL_SIZE
            # Exit if Alligator goes to sleep or reverses
            if alligator_direction_aligned[i] != -1:
                signals[i] = 0.0
                position = 0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8515: 6h Williams Alligator + Elder Ray + ADX Regime
Hypothesis: The Williams Alligator (jaw/teeth/lips) identifies trend direction and strength, 
while Elder Ray (bull/bear power) measures trend momentum. Combined with ADX regime filtering,
this creates a robust trend-following system that works in both bull and bear markets by 
only trading when the Alligator is 'awake' (trending) and ADX confirms strong trend.
The Elder Ray filter ensures we only take trades in the direction of underlying power,
reducing false signals during weak trends. Targets 50-150 total trades over 4 years (12-37/year)
to minimize fee drag while maintaining statistical validity.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8515_6h_alligator_elder_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13  # Smoothed with 8-period shift
ALLIGATOR_PERIOD_TEETH = 8  # Smoothed with 5-period shift
ALLIGATOR_PERIOD_LIPS = 5   # Smoothed with 3-period shift
ELDER_RAY_PERIOD = 13       # EMA for Bull/Bear Power calculation
ADX_PERIOD = 14             # Standard ADX period
ADX_THRESHOLD = 25          # Minimum ADX for trending market
EMA_FAST = 12               # For EMA calculation in Elder Ray
EMA_SLOW = 26               # For EMA calculation in Elder Ray
SIGNAL_SIZE = 0.25          # Position size (25% of capital)

def calculate_wma(series, period):
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return np.convolve(series, weights / weights.sum(), mode='same')

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines (Jaw, Teeth, Lips)"""
    # Median price
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).ewm(alpha=1/ALLIGATOR_PERIOD_JAW, adjust=False, min_periods=ALLIGATOR_PERIOD_JAW).mean().values
    jaw = np.roll(jaw, -ALLIGATOR_PERIOD_JAW//2)  # Center the smoothed line
    
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).ewm(alpha=1/ALLIGATOR_PERIOD_TEETH, adjust=False, min_periods=ALLIGATOR_PERIOD_TEETH).mean().values
    teeth = np.roll(teeth, -ALLIGATOR_PERIOD_TEETH//2)
    
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).ewm(alpha=1/ALLIGATOR_PERIOD_LIPS, adjust=False, min_periods=ALLIGATOR_PERIOD_LIPS).mean().values
    lips = np.roll(lips, -ALLIGATOR_PERIOD_LIPS//2)
    
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close):
    """Calculate Elder Ray: Bull Power and Bear Power"""
    # Calculate 13-period EMA
    ema = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    
    # Bull Power = High - EMA
    bull_power = high - ema
    
    # Bear Power = Low - EMA
    bear_power = low - ema
    
    return bull_power, bear_power

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_period
    di_minus = 100 * dm_minus_smooth / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams Alligator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high_1d, low_1d, close_1d)
    
    # Alligator is 'awake' when lips > teeth > jaw (uptrend) or lips < teeth < jaw (downtrend)
    alligator_up = (lips_1d > teeth_1d) & (teeth_1d > jaw_1d)
    alligator_down = (lips_1d < teeth_1d) & (teeth_1d < jaw_1d)
    alligator_awake = alligator_up | alligator_down
    
    # Determine trend direction from Alligator
    alligator_direction = np.where(alligator_up, 1, np.where(alligator_down, -1, 0))
    alligator_direction_aligned = align_htf_to_ltf(prices, df_1d, alligator_direction)
    
    # Calculate 1d Elder Ray
    bull_power_1d, bear_power_1d = calculate_elder_ray(high_1d, low_1d, close_1d)
    
    # Elder Ray signals: Bull Power > 0 and rising, Bear Power < 0 and falling
    bull_power_rising = bull_power_1d > np.roll(bull_power_1d, 1)
    bear_power_falling = bear_power_1d < np.roll(bear_power_1d, 1)
    elder_bull = bull_power_1d > 0 & bull_power_rising
    elder_bear = bear_power_1d < 0 & bear_power_falling
    
    # Elder Ray direction
    elder_direction = np.where(elder_bull, 1, np.where(elder_bear, -1, 0))
    elder_direction_aligned = align_htf_to_ltf(prices, df_1d, elder_direction)
    
    # Calculate 1d ADX for regime filtering
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    strong_trend = adx_1d >= ADX_THRESHOLD
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator on 6h for entry timing
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Entry signals based on Alligator crossover
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # Bullish crossover: lips crosses above teeth
    bullish_crossover = lips_above_teeth & ~np.roll(lips_above_teeth, 1)
    # Bearish crossover: lips crosses below teeth
    bearish_crossover = lips_below_teeth & ~np.roll(lips_below_teeth, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS,
                ELDER_RAY_PERIOD, ADX_PERIOD) + 10
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(alligator_direction_aligned[i]) or np.isnan(elder_direction_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Check for trend reversal (Alligator sleeping)
        if alligator_direction_aligned[i] == 0:  # Alligator sleeping
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trade direction based on HTF alignment
        htf_alligator_dir = alligator_direction_aligned[i]
        htf_elder_dir = elder_direction_aligned[i]
        htf_strong_trend = strong_trend_aligned[i]
        
        # Require alignment between Alligator direction and Elder Ray
        direction_aligned = (htf_alligator_dir == htf_elder_dir) and (htf_alligator_dir != 0)
        
        # Only trade in strong trend regimes
        if not (direction_aligned and htf_strong_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check for entry signals on 6h timeframe
        if position == 0:
            # Long entry: bullish crossover in alignment with bullish HTF
            if htf_alligator_dir == 1 and bullish_crossover[i]:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            # Short entry: bearish crossover in alignment with bearish HTF
            elif htf_alligator_dir == -1 and bearish_crossover[i]:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Maintain long position
            signals[i] = SIGNAL_SIZE
            # Exit if Alligator goes to sleep or reverses
            if alligator_direction_aligned[i] != 1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Maintain short position
            signals[i] = -SIGNAL_SIZE
            # Exit if Alligator goes to sleep or reverses
            if alligator_direction_aligned[i] != -1:
                signals[i] = 0.0
                position = 0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8515: 6h Donchian Breakout + Weekly Pivot + Volume Confirmation
Hypothesis: Combines 6-hour Donchian(20) breakouts with weekly pivot point direction 
(from 1w timeframe) and volume confirmation to filter for institutional participation.
Weekly pivots provide key support/resistance levels that price tends to respect, 
while Donchian breakouts capture momentum. Volume confirmation ensures breakouts 
have sufficient conviction. This approach works in both bull and bear markets by 
only taking breakouts in the direction of the weekly pivot bias, reducing false 
signals during ranging periods. Targets 50-150 total trades over 4 years (12-37/year)
to balance statistical significance with minimal fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8515_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5  # Number of weeks to calculate pivot from
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w, r1_1w, s1_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Determine weekly bias: price above pivot = bullish, below = bearish
    weekly_bias = np.where(close_1w > pivot_1w, 1, np.where(close_1w < pivot_1w, -1, 0))
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD) + 5
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_bias_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Determine trade direction from weekly bias
        bullish_bias = weekly_bias_aligned[i] == 1
        bearish_bias = weekly_bias_aligned[i] == -1
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]
        short_breakout = close[i] < donchian_low[i-1]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions - only trade in direction of weekly bias
        long_entry = bullish_bias and long_breakout and volume_confirmed
        short_entry = bearish_bias and short_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Maintain long position
            signals[i] = SIGNAL_SIZE
            # Exit if price breaks below weekly support (S1) or Donchian low
            if close[i] < s1_1w[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Maintain short position
            signals[i] = -SIGNAL_SIZE
            # Exit if price breaks above weekly resistance (R1) or Donchian high
            if close[i] > r1_1w[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8515: 6h Donchian Breakout + Weekly Pivot + Volume Confirmation
Hypothesis: Combines 6-hour Donchian(20) breakouts with weekly pivot point direction 
(from 1w timeframe) and volume confirmation to filter for institutional participation.
Weekly pivots provide key support/resistance levels that price tends to respect, 
while Donchian breakouts capture momentum. Volume confirmation ensures breakouts 
have sufficient conviction. This approach works in both bull and bear markets by 
only taking breakouts in the direction of the weekly pivot bias, reducing false 
signals during ranging periods. Targets 50-150 total trades over 4 years (12-37/year)
to balance statistical significance with minimal fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8515_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5  # Number of weeks to calculate pivot from
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w, r1_1w, s1_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Determine weekly bias: price above pivot = bullish, below = bearish
    weekly_bias = np.where(close_1w > pivot_1w, 1, np.where(close_1w < pivot_1w, -1, 0))
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD) + 5