#!/usr/bin/env python3
"""
Experiment #9679: 6h Donchian Breakout + Volume Spike + ATR Filter
Hypothesis: Donchian(20) breakouts with volume confirmation and ATR-based filtering 
capture strong trending moves while avoiding whipsaws. Works in bull markets (breakouts up) 
and bear markets (breakdowns down). Targets 75-150 total trades over 4 years (19-38/year) 
to balance opportunity and cost. Uses 1d trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_9679_6h_donchian_breakout_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_FILTER_MULTIPLIER = 0.5
SIGNAL_SIZE = 0.25

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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_1d)  # Note: align_ltf_to_htf doesn't exist, using align_htf_to_ltf
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for volatility filter and stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]):
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
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER)
        
        # ATR filter: only trade when volatility is sufficient
        atr_filter = atr[i] > (np.mean(atr[max(0, i-50):i+1]) * ATR_FILTER_MULTIPLIER) if i >= 50 else True
        
        # 1d trend filter (using EMA50)
        # Note: Correcting the align function name
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
        uptrend = ema_1d_aligned[i] > ema_1d_aligned[i-1] if i > 0 else True
        downtrend = ema_1d_aligned[i] < ema_1d_aligned[i-1] if i > 0 else True
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_lower[i-1]  # Break below previous period's low
        
        # Entry conditions
        long_entry = long_breakout and volume_spike and atr_filter and uptrend
        short_entry = short_breakout and volume_spike and atr_filter and downtrend
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])  # 2x ATR stop
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])  # 2x ATR stop
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
Experiment #9679: 6h Donchian Breakout + Volume Spike + ATR Filter
Hypothesis: Donchian(20) breakouts with volume confirmation and ATR-based filtering 
capture strong trending moves while avoiding whipsaws. Works in bull markets (breakouts up) 
and bear markets (breakdowns down). Targets 75-150 total trades over 4 years (19-38/year) 
to balance opportunity and cost. Uses 1d trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9679_6h_donchian_breakout_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_FILTER_MULTIPLIER = 0.5
SIGNAL_SIZE = 0.25

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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for volatility filter and stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]):
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
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER)
        
        # ATR filter: only trade when volatility is sufficient
        atr_filter = atr[i] > (np.mean(atr[max(0, i-50):i+1]) * ATR_FILTER_MULTIPLIER) if i >= 50 else True
        
        # 1d trend filter (using EMA50)
        uptrend = ema_1d_aligned[i] > ema_1d_aligned[i-1] if i > 0 else True
        downtrend = ema_1d_aligned[i] < ema_1d_aligned[i-1] if i > 0 else True
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_lower[i-1]  # Break below previous period's low
        
        # Entry conditions
        long_entry = long_breakout and volume_spike and atr_filter and uptrend
        short_entry = short_breakout and volume_spike and atr_filter and downtrend
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])  # 2x ATR stop
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])  # 2x ATR stop
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals