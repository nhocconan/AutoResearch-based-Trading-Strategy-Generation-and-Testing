#!/usr/bin/env python3
"""
Experiment #9664: 1d Donchian Breakout + Volume Spike + Weekly Trend Filter.
Hypothesis: Donchian(20) breakouts on the 1d timeframe, filtered by weekly trend direction and volume spikes,
capture strong directional moves while avoiding false breakouts in ranging markets. Weekly trend filter ensures
alignment with higher timeframe momentum, reducing whipsaws. Designed for 1d timeframe with target of 30-100
total trades over 4 years (7-25/year) to minimize fee drag and improve generalization to bear markets (2025+).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9664_1d_donchian_breakout_volume_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
WEEKLY_TREND_PERIOD = 50  # EMA for weekly trend
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    weekly_ema = pd.Series(close_weekly).ewm(span=WEEKLY_TREND_PERIOD, adjust=False, min_periods=WEEKLY_TREND_PERIOD).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate 1d Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper and lower bands
    donchian_upper = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_lower = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_TREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if weekly trend data not available
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter: price above/below weekly EMA
        uptrend = close[i] > weekly_ema_aligned[i]
        downtrend = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        upper_breakout = close[i] >= donchian_upper[i]
        lower_breakout = close[i] <= donchian_lower[i]
        
        # Entry conditions
        long_entry = uptrend and volume_spike and upper_breakout
        short_entry = downtrend and volume_spike and lower_breakout
        
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
</assistant>assistant</think>
#!/usr/bin/env python3
"""
Experiment #9664: 1d Donchian Breakout + Volume Spike + Weekly Trend Filter.
Hypothesis: Donchian(20) breakouts on the 1d timeframe, filtered by weekly trend direction and volume spikes,
capture strong directional moves while avoiding false breakouts in ranging markets. Weekly trend filter ensures
alignment with higher timeframe momentum, reducing whipsaws. Designed for 1d timeframe with target of 30-100
total trades over 4 years (7-25/year) to minimize fee drag and improve generalization to bear markets (2025+).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9664_1d_donchian_breakout_volume_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
WEEKLY_TREND_PERIOD = 50  # EMA for weekly trend
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    weekly_ema = pd.Series(close_weekly).ewm(span=WEEKLY_TREND_PERIOD, adjust=False, min_periods=WEEKLY_TREND_PERIOD).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate 1d Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper and lower bands
    donchian_upper = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_lower = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_TREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if weekly trend data not available
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter: price above/below weekly EMA
        uptrend = close[i] > weekly_ema_aligned[i]
        downtrend = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        upper_breakout = close[i] >= donchian_upper[i]
        lower_breakout = close[i] <= donchian_lower[i]
        
        # Entry conditions
        long_entry = uptrend and volume_spike and upper_breakout
        short_entry = downtrend and volume_spike and lower_breakout
        
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
</assistant>assistant is done.