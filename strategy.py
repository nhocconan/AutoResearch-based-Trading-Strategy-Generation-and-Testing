#!/usr/bin/env python3
"""
Experiment #9550: 1d Donchian Breakout + Volume + 1w Trend Filter
Hypothesis: 1d Donchian(20) breakouts with volume confirmation and 1w EMA trend filter
will capture major moves in both bull and bear markets. Weekly EMA filter ensures
we only trade in the direction of the higher timeframe trend, reducing whipsaws.
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9550_1d_donchian20_volume_1wema_trend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
EMA_FAST = 9
EMA_SLOW = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_fast_1w = calculate_ema(close_1w, EMA_FAST)
    ema_slow_1w = calculate_ema(close_1w, EMA_SLOW)
    ema_fast_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_fast_1w)
    ema_slow_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slow_1w)
    
    # Calculate LTF indicators (1d)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
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
    start = max(DONCHIAN_PERIOD, 20, EMA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_fast_1w_aligned[i]) or np.isnan(ema_slow_1w_aligned[i]):
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
        
        # 1w Trend filter: EMA9 > EMA21 = uptrend, EMA9 < EMA21 = downtrend
        uptrend = ema_fast_1w_aligned[i] > ema_slow_1w_aligned[i]
        downtrend = ema_fast_1w_aligned[i] < ema_slow_1w_aligned[i]
        
        # Entry conditions: Donchian breakout + volume + trend filter
        long_entry = (close[i] >= donchian_upper[i]) and volume_spike and uptrend
        short_entry = (close[i] <= donchian_lower[i]) and volume_spike and downtrend
        
        # Exit conditions: reverse signal or stoploss (handled above)
        long_exit = close[i] <= donchian_lower[i]  # Exit long if price hits lower band
        short_exit = close[i] >= donchian_upper[i]  # Exit short if price hits upper band
        
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
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #9550: 1d Donchian Breakout + Volume + 1w Trend Filter
Hypothesis: 1d Donchian(20) breakouts with volume confirmation and 1w EMA trend filter
will capture major moves in both bull and bear markets. Weekly EMA filter ensures
we only trade in the direction of the higher timeframe trend, reducing whipsaws.
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9550_1d_donchian20_volume_1wema_trend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
EMA_FAST = 9
EMA_SLOW = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_fast_1w = calculate_ema(close_1w, EMA_FAST)
    ema_slow_1w = calculate_ema(close_1w, EMA_SLOW)
    ema_fast_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_fast_1w)
    ema_slow_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slow_1w)
    
    # Calculate LTF indicators (1d)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
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
    start = max(DONCHIAN_PERIOD, 20, EMA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_fast_1w_aligned[i]) or np.isnan(ema_slow_1w_aligned[i]):
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
        
        # 1w Trend filter: EMA9 > EMA21 = uptrend, EMA9 < EMA21 = downtrend
        uptrend = ema_fast_1w_aligned[i] > ema_slow_1w_aligned[i]
        downtrend = ema_fast_1w_aligned[i] < ema_slow_1w_aligned[i]
        
        # Entry conditions: Donchian breakout + volume + trend filter
        long_entry = (close[i] >= donchian_upper[i]) and volume_spike and uptrend
        short_entry = (close[i] <= donchian_lower[i]) and volume_spike and downtrend
        
        # Exit conditions: reverse signal or stoploss (handled above)
        long_exit = close[i] <= donchian_lower[i]  # Exit long if price hits lower band
        short_exit = close[i] >= donchian_upper[i]  # Exit short if price hits upper band
        
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
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>