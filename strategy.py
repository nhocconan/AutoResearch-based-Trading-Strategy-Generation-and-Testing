#!/usr/bin/env python3
"""
Experiment #9807: 6h Donchian Breakout + Daily Pivot Reversal + Volume Confirmation
Hypothesis: Fade at daily S3/R3 pivot levels with Donchian breakout confirmation and volume spike
works in both bull and bear markets by capturing mean reversion at key levels with momentum confirmation.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9807_6h_donchian_breakout_daily_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 14
PIVOT_LOOKBACK = 20  # Days for pivot calculation
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, S1, R2, S2, R3, S3"""
    p = (high + low + close) / 3.0
    r1 = 2*p - low
    s1 = 2*p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2*(p - low)
    s3 = low - 2*(high - p)
    return p, r1, s1, r2, s2, r3, s3

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot points
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Initialize pivot arrays
    r3 = np.full_like(daily_close, np.nan)
    s3 = np.full_like(daily_close, np.nan)
    
    # Calculate pivots for each day
    for i in range(len(daily_close)):
        if i >= PIVOT_LOOKBACK:
            # Use lookback period for pivot calculation (more stable)
            lookback_start = max(0, i - PIVOT_LOOKBACK + 1)
            lookback_end = i + 1
            period_high = np.max(daily_high[lookback_start:lookback_end])
            period_low = np.min(daily_low[lookback_start:lookback_end])
            period_close = daily_close[i]  # Today's close
            
            _, _, _, _, _, r3[i], s3[i] = calculate_pivot_points(period_high, period_low, period_close)
    
    # Align daily pivots to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, 20) + 1
    
    for i in range(start, n):
        # Skip if pivots not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Pivot rejection conditions
        near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < 0.005  # Within 0.5% of S3
        near_r3 = abs(close[i] - r3_aligned[i]) / r3_aligned[i] < 0.005  # Within 0.5% of R3
        
        # Donchian breakout conditions
        donch_breakout_up = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        donch_breakout_down = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: pivot rejection with Donchian breakout in opposite direction
        # Long: price near S3 + breaks above Donchian upper (bullish rejection)
        long_entry = near_s3 and donch_breakout_up and volume_spike
        # Short: price near R3 + breaks below Donchian lower (bearish rejection)
        short_entry = near_r3 and donch_breakout_down and volume_spike
        
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
Experiment #9807: 6h Donchian Breakout + Daily Pivot Reversal + Volume Confirmation
Hypothesis: Fade at daily S3/R3 pivot levels with Donchian breakout confirmation and volume spike
works in both bull and bear markets by capturing mean reversion at key levels with momentum confirmation.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9807_6h_donchian_breakout_daily_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 14
PIVOT_LOOKBACK = 20  # Days for pivot calculation
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, S1, R2, S2, R3, S3"""
    p = (high + low + close) / 3.0
    r1 = 2*p - low
    s1 = 2*p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2*(p - low)
    s3 = low - 2*(high - p)
    return p, r1, s1, r2, s2, r3, s3

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot points
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Initialize pivot arrays
    r3 = np.full_like(daily_close, np.nan)
    s3 = np.full_like(daily_close, np.nan)
    
    # Calculate pivots for each day
    for i in range(len(daily_close)):
        if i >= PIVOT_LOOKBACK:
            # Use lookback period for pivot calculation (more stable)
            lookback_start = max(0, i - PIVOT_LOOKBACK + 1)
            lookback_end = i + 1
            period_high = np.max(daily_high[lookback_start:lookback_end])
            period_low = np.min(daily_low[lookback_start:lookback_end])
            period_close = daily_close[i]  # Today's close
            
            _, _, _, _, _, r3[i], s3[i] = calculate_pivot_points(period_high, period_low, period_close)
    
    # Align daily pivots to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, 20) + 1
    
    for i in range(start, n):
        # Skip if pivots not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Pivot rejection conditions
        near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < 0.005  # Within 0.5% of S3
        near_r3 = abs(close[i] - r3_aligned[i]) / r3_aligned[i] < 0.005  # Within 0.5% of R3
        
        # Donchian breakout conditions
        donch_breakout_up = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        donch_breakout_down = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: pivot rejection with Donchian breakout in opposite direction
        # Long: price near S3 + breaks above Donchian upper (bullish rejection)
        long_entry = near_s3 and donch_breakout_up and volume_spike
        # Short: price near R3 + breaks below Donchian lower (bearish rejection)
        short_entry = near_r3 and donch_breakout_down and volume_spike
        
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