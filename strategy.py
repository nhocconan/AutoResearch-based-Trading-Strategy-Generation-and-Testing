#!/usr/bin/env python3
"""
Experiment #9575: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Donchian(20) breakouts on 6h chart aligned with weekly pivot bias (price above/below weekly pivot)
and volume confirmation yield high-probability trend continuation trades. Weekly pivot provides structural bias
(above pivot = bullish bias, below = bearish) to filter breakouts, reducing false signals in chop.
Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Uses discrete position sizing (0.25) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9575_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_PIVOT_METHOD = 'standard'  # (H+L+C)/3
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]):
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
        
        # Breakout conditions with weekly pivot bias
        bullish_bias = close[i] > weekly_pivot_aligned[i]  # Price above weekly pivot = bullish bias
        bearish_bias = close[i] < weekly_pivot_aligned[i]  # Price below weekly pivot = bearish bias
        
        # Long: Donchian breakout + bullish bias + volume spike
        long_breakout = close[i] > donchian_upper[i]
        long_entry = long_breakout and bullish_bias and volume_spike
        
        # Short: Donchian breakdown + bearish bias + volume spike
        short_breakout = close[i] < donchian_lower[i]
        short_entry = short_breakout and bearish_bias and volume_spike
        
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
Experiment #9575: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Donchian(20) breakouts on 6h chart aligned with weekly pivot bias (price above/below weekly pivot)
and volume confirmation yield high-probability trend continuation trades. Weekly pivot provides structural bias
(above pivot = bullish bias, below = bearish) to filter breakouts, reducing false signals in chop.
Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Uses discrete position sizing (0.25) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9575_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_PIVOT_METHOD = 'standard'  # (H+L+C)/3
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]):
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
        
        # Breakout conditions with weekly pivot bias
        bullish_bias = close[i] > weekly_pivot_aligned[i]  # Price above weekly pivot = bullish bias
        bearish_bias = close[i] < weekly_pivot_aligned[i]  # Price below weekly pivot = bearish bias
        
        # Long: Donchian breakout + bullish bias + volume spike
        long_breakout = close[i] > donchian_upper[i]
        long_entry = long_breakout and bullish_bias and volume_spike
        
        # Short: Donchian breakdown + bearish bias + volume spike
        short_breakout = close[i] < donchian_lower[i]
        short_entry = short_breakout and bearish_bias and volume_spike
        
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