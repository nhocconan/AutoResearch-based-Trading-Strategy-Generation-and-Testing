#!/usr/bin/env python3
"""
Experiment #9899: 6h Williams Alligator + Elder Ray + Weekly Trend
Hypothesis: Combines Williams Alligator (trend detection) with Elder Ray (bull/bear power) and weekly trend filter to capture strong trends while avoiding whipsaws. Works in bull markets (bullish alignment above weekly EMA) and bear markets (bearish alignment below weekly EMA). Williams Alligator filters sideways markets, Elder Ray confirms momentum. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9899_6h_williams_alligator_elder_ray_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
WEEKLY_EMA_PERIOD = 40
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(high, low, close, jaw_period, teeth_period, lips_period):
    """Calculate Williams Alligator lines"""
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=jaw_period, center=True).mean().shift(jaw_period//2).values
    teeth = pd.Series(median_price).rolling(window=teeth_period, center=True).mean().shift(teeth_period//2).values
    lips = pd.Series(median_price).rolling(window=lips_period, center=True).mean().shift(lips_period//2).values
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
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, WEEKLY_EMA_PERIOD)
    
    # Align weekly EMA to 6h timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close, 
                                          ALLIGATOR_JAW_PERIOD,
                                          ALLIGATOR_TEETH_PERIOD,
                                          ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD,
                ELDER_RAY_PERIOD, WEEKLY_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
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
        
        # Williams Alligator conditions: aligned for trend
        bullish_alignment = (lips[i] > teeth[i] > jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        bearish_alignment = (lips[i] < teeth[i] < jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        
        # Elder Ray conditions: momentum confirmation
        bullish_momentum = bull_power[i] > 0 if not np.isnan(bull_power[i]) else False
        bearish_momentum = bear_power[i] < 0 if not np.isnan(bear_power[i]) else False
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        # Entry conditions
        long_entry = bullish_alignment and bullish_momentum and above_weekly_ema
        short_entry = bearish_alignment and bearish_momentum and below_weekly_ema
        
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
Experiment #9899: 6h Donchian Breakout + 12h Supertrend + Volume Spike
Hypothesis: Donchian(20) breakouts filtered by 12h Supertrend direction and volume confirmation. Works in bull markets (breakouts above Supertrend) and bear markets (breakdowns below Supertrend). Volume reduces false breakouts. Supertrend filters sideways markets. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9899_6h_donchian_breakout_12h_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    hl2 = (high + low) / 2
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Basic bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize final bands
    final_upper = np.full_like(upper_band, np.nan)
    final_lower = np.full_like(lower_band, np.nan)
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(len(close)):
        if i == 0:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
            supertrend[i] = upper_band[i]
            direction[i] = -1  # start in downtrend
        else:
            # Upper band logic
            if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            # Lower band logic
            if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            # Supertrend and direction
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
            else:
                if close[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
    
    return supertrend, direction

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
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
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Supertrend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Supertrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    supertrend_12h, direction_12h = calculate_supertrend(high_12h, low_12h, close_12h, 
                                                         SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    
    # Align 12h Supertrend direction to 6h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, SUPERTREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if 12h Supertrend not available
        if np.isnan(supertrend_dir_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: Supertrend direction from 12h
        uptrend = supertrend_dir_aligned[i] > 0
        downtrend = supertrend_dir_aligned[i] < 0
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of 12h Supertrend with volume
        long_entry = bullish_breakout and uptrend and volume_spike
        short_entry = bearish_breakout and downtrend and volume_spike
        
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
Experiment #9899: 6h Camarilla Pivot + 1d ADX + Volume Spike
Hypothesis: Camarilla pivot levels from daily timeframe provide precise entry/exit points. Fade at R3/S3 (mean reversion in range), breakout continuation at R4/S4 (trend following). ADX(1d) filters for trending markets (ADX>25). Volume confirms breakouts. Works in bull markets (R4 breakouts) and bear markets (S4 breakdowns). Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9899_6h_camarilla_pivot_1d_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_adx(high, low, close, period):
    """Calculate ADX indicator"""
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
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_hl = high - low
    r1 = close + (range_hl * 1.1 / 12)
    r2 = close + (range_hl * 1.1 / 6)
    r3 = close + (range_hl * 1.1 / 4)
    r4 = close + (range_hl * 1.1 / 2)
    s1 = close - (range_hl * 1.1 / 12)
    s2 = close - (range_hl * 1.1 / 6)
    s3 = close - (range_hl * 1.1 / 4)
    s4 = close - (range_hl * 1.1 / 2)
    return r1, r2, r3, r4, s1, s2, s3, s4

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
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate daily ADX for trend filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(adx_aligned[i]):
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
        
        # Volume spike confirmation
        volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # ADX trend filter
        trending = adx_aligned[i] > ADX_THRESHOLD
        
        # Camarilla levels
        r3_level = r3_aligned[i]
        r4_level = r4_aligned[i]
        s3_level = s3_aligned[i]
        s4_level = s4_aligned[i]
        
        # Entry conditions
        # Long: break above R4 in trending market OR bounce from S3 in ranging market
        long_breakout = close[i] > r4_level and trending and volume_spike
        long_bounce = close[i] > s3_level and close[i-1] <= s3_level and not trending and volume_spike
        
        # Short: break below S4 in trending market OR bounce from R3 in ranging market
        short_breakout = close[i] < s4_level and trending and volume_spike
        short_bounce = close[i] < r3_level and close[i-1] >= r3_level and not trending and volume_spike
        
        long_entry = long_breakout or long_bounce
        short_entry = short_breakout or short_bounce
        
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
Experiment #9899: 6h Elder Ray + 12h Fisher Transform + Volume Spike
Hypothesis: Combines Elder Ray (bull/bear power) with Ehlers Fisher Transform for precise turning points. Fisher Transform identifies extreme price movements likely to reverse. Elder Ray confirms momentum direction. Volume spike validates breakouts. Works in bull markets (bullish extreme + buying power) and bear markets (bearish extreme + selling power). Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9899_6h_elder_ray_12h_fisher_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
FISHER_PERIOD = 10
FISHER_THRESHOLD = 1.5
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def calculate_fisher_transform(high, low, period):
    """Calculate Ehlers Fisher Transform"""
    # Median price
    median_price = (high + low) / 2
    
    # Normalize to [-1, 1] range over period
    highest = pd.Series(median_price).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(median_price).rolling(window=period, min_periods=period).min().values
    range_val = highest - lowest
    
    # Avoid division by zero
    value = np.where(range_val != 0, 
                     2 * ((median_price - lowest) / range_val - 0.5), 
                     0)
    
    # Smooth the value
    smoothed = pd.Series(value).ewm(alpha=0.5, adjust=False).mean().values
    
    # Fisher Transform
    # Clamp to avoid log domain issues
    smoothed_clamped = np.clip(smoothed, -0.999, 0.999)
    fisher = 0.5 * np.log((1 + smoothed_clamped) / (1 - smoothed_clamped))
    
    return fisher

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
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Fisher Transform
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Fisher Transform
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    fisher_12h = calculate_fisher_transform(high_12h, low_12h, FISHER_PERIOD)
    
    # Align 12h Fisher Transform to 6h timeframe
    fisher_aligned = align_htf_to_ltf(prices, df_12h, fisher_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, FISHER_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h Fisher not available
        if np.isnan(fisher_aligned[i]):
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
        
        # Volume spike confirmation
        volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Fisher Transform extremes: > threshold = overbought, < -threshold = oversold
        fisher_overbought = fisher_aligned[i] > FISHER_THRESHOLD
        fisher_oversold = fisher_aligned[i] < -FISHER_THRESHOLD
        
        # Elder Ray momentum
        bullish_momentum = bull_power[i] > 0 if not np.isnan(bull_power[i]) else False
        bearish_momentum = bear_power[i] < 0 if not np.isnan(bear_power[i]) else False
        
        # Entry conditions: Fisher extreme + Elder Ray momentum + volume
        # Long: oversold Fisher + bullish momentum
        long_entry = fisher_oversold and bullish_momentum and volume_spike
        # Short: overbought Fisher + bearish momentum
        short_entry = fisher_overbought and bearish_momentum and volume_spike
        
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
Experiment #9899: 6h Connors RSI + 1d Trend + Volume Spike
Hypothesis: Connors RSI (RSI(3) + RSI Streak + PercentRank) identifies extreme mean-reversion points. Trades in direction of 1d trend (EMA50) with volume confirmation. Works in bull markets (pullbacks to RSI<10 in uptrend) and bear markets (bounces to RSI>90 in downtrend). Volume filters false signals. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9899_6h_connors_rsi_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
RSI_PERIOD = 3
STREAK_PERIOD = 2
PERCENT_RANK_LOOKBACK = 100
DAILY_EMA_PERIOD = 50
V