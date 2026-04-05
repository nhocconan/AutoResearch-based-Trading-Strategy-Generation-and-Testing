#!/usr/bin/env python3
"""
Experiment #8395: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: On 6h timeframe, buy when price breaks above Donchian(20) high with weekly pivot support and volume confirmation,
sell when price breaks below Donchian(20) low with weekly pivot resistance and volume confirmation. Weekly pivot provides
institutional bias, Donchian captures breakouts, volume filters false breakouts. Targets 75-150 trades over 4 years.
Works in bull via breakouts, in bear via short breakdowns with pivot resistance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8395_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P, R1, R2, R3, S1, S2, S3"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper, lower"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    pivot, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Use S3 as support, R3 as resistance for breakout context
    support_level = s3
    resistance_level = r3
    
    # Align weekly levels to 6h timeframe
    support_aligned = align_htf_to_ltf(prices, df_1w, support_level)
    resistance_aligned = align_htf_to_ltf(prices, df_1w, resistance_level)
    
    # Calculate 6h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(support_aligned[i]) or np.isnan(resistance_aligned[i]):
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
        
        # Determine if weekly pivot provides support/resistance context
        price_above_support = close[i] > support_aligned[i]
        price_below_resistance = close[i] < resistance_aligned[i]
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        bearish_breakdown = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: breakout with pivot context and volume
        long_entry = bullish_breakout and price_above_support and volume_confirmed
        short_entry = bearish_breakdown and price_below_resistance and volume_confirmed
        
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
Experiment #8395: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: On 6h timeframe, buy when price breaks above Donchian(20) high with weekly pivot support and volume confirmation,
sell when price breaks below Donchian(20) low with weekly pivot resistance and volume confirmation. Weekly pivot provides
institutional bias, Donchian captures breakouts, volume filters false breakouts. Targets 75-150 trades over 4 years.
Works in bull via breakouts, in bear via short breakdowns with pivot resistance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8395_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P, R1, R2, R3, S1, S2, S3"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper, lower"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    pivot, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Use S3 as support, R3 as resistance for breakout context
    support_level = s3
    resistance_level = r3
    
    # Align weekly levels to 6h timeframe
    support_aligned = align_htf_to_ltf(prices, df_1w, support_level)
    resistance_aligned = align_htf_to_ltf(prices, df_1w, resistance_level)
    
    # Calculate 6h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(support_aligned[i]) or np.isnan(resistance_aligned[i]):
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
        
        # Determine if weekly pivot provides support/resistance context
        price_above_support = close[i] > support_aligned[i]
        price_below_resistance = close[i] < resistance_aligned[i]
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        bearish_breakdown = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: breakout with pivot context and volume
        long_entry = bullish_breakout and price_above_support and volume_confirmed
        short_entry = bearish_breakdown and price_below_resistance and volume_confirmed
        
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