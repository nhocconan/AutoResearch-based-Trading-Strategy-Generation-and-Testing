#!/usr/bin/env python3
"""
Experiment #9690: 1d Donchian Breakout with Weekly Trend + Volume Confirmation
Hypothesis: Donchian channel breakouts on the daily timeframe, filtered by weekly trend direction 
and confirmed by volume spikes, capture significant price moves while avoiding whipsaws. 
The weekly trend filter ensures we only trade in the direction of the higher timeframe trend, 
reducing losses during counter-trend moves. Volume confirmation adds validity to breakouts. 
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee impact.
Works in bull markets (breakouts with upward weekly trend) and bear markets (breakouts with downward weekly trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9690_1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
WEEKLY_TREND_PERIOD = 50  # EMA for weekly trend
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
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
    weekly_ema = calculate_ema(weekly_close, WEEKLY_TREND_PERIOD)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
    start = max(DONCHIAN_PERIOD, WEEKLY_TREND_PERIOD, ATR_PERIOD) + 1
    
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
        
        # Weekly trend direction
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper[i]
        bearish_breakout = close[i] < donchian_lower[i]
        
        # Entry conditions: breakout in direction of weekly trend with volume confirmation
        long_entry = bullish_breakout and weekly_uptrend and volume_spike
        short_entry = bearish_breakout and weekly_downtrend and volume_spike
        
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
Experiment #9690: 1d Donchian Breakout with Weekly Trend + Volume Confirmation
Hypothesis: Donchian channel breakouts on the daily timeframe, filtered by weekly trend direction 
and confirmed by volume spikes, capture significant price moves while avoiding whipsaws. 
The weekly trend filter ensures we only trade in the direction of the higher timeframe trend, 
reducing losses during counter-trend moves. Volume confirmation adds validity to breakouts. 
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee impact.
Works in bull markets (breakouts with upward weekly trend) and bear markets (breakouts with downward weekly trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9690_1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
WEEKLY_TREND_PERIOD = 50  # EMA for weekly trend
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
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
    weekly_ema = calculate_ema(weekly_close, WEEKLY_TREND_PERIOD)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
    start = max(DONCHIAN_PERIOD, WEEKLY_TREND_PERIOD, ATR_PERIOD) + 1
    
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
        
        # Weekly trend direction
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper[i]
        bearish_breakout = close[i] < donchian_lower[i]
        
        # Entry conditions: breakout in direction of weekly trend with volume confirmation
        long_entry = bullish_breakout and weekly_uptrend and volume_spike
        short_entry = bearish_breakout and weekly_downtrend and volume_spike
        
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
Experiment #9690: 1d Donchian Breakout with Weekly Trend + Volume Confirmation
Hypothesis: Donchian channel breakouts on the daily timeframe, filtered by weekly trend direction 
and confirmed by volume spikes, capture significant price moves while avoiding whipsaws. 
The weekly trend filter ensures we only trade in the direction of the higher timeframe trend, 
reducing losses during counter-trend moves. Volume confirmation adds validity to breakouts. 
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee impact.
Works in bull markets (breakouts with upward weekly trend) and bear markets (breakouts with downward weekly trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9690_1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
WEEKLY_TREND_PERIOD = 50  # EMA for weekly trend
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
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
    weekly_ema = calculate_ema(weekly_close, WEEKLY_TREND_PERIOD)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
    start = max(DONCHIAN_PERIOD, WEEKLY_TREND_PERIOD, ATR_PERIOD) + 1
    
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
        
        # Weekly trend direction
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper[i]
        bearish_breakout = close[i] < donchian_lower[i]
        
        # Entry conditions: breakout in direction of weekly trend with volume confirmation
        long_entry = bullish_breakout and weekly_uptrend and volume_spike
        short_entry = bearish_breakout and weekly_downtrend and volume_spike
        
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

</script>