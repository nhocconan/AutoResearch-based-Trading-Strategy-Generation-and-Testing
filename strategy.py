#!/usr/bin/env python3
"""
Experiment #9658: 1d Donchian Breakout + Volume Spike + 1w Trend Filter.
Hypothesis: 1d Donchian(20) breakouts combined with volume spikes and 1w trend filter
provide high-probability entries in both bull and bear markets. The 1w trend ensures
we only trade in the direction of the higher timeframe trend, reducing whipsaw.
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9658_1d_donchian_breakout_volume_1wtrend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
TREND_PERIOD = 50  # For 1w EMA trend filter
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
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
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
        
        # Trend filter: only long in uptrend, only short in downtrend
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals with volume confirmation and trend filter
        breakout_long = volume_spike and uptrend and close[i] >= donchian_high[i]
        breakout_short = volume_spike and downtrend and close[i] <= donchian_low[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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
Experiment #9658: 1d Donchian Breakout + Volume Spike + 1w Trend Filter.
Hypothesis: 1d Donchian(20) breakouts combined with volume spikes and 1w trend filter
provide high-probability entries in both bull and bear markets. The 1w trend ensures
we only trade in the direction of the higher timeframe trend, reducing whipsaw.
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9658_1d_donchian_breakout_volume_1wtrend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
TREND_PERIOD = 50  # For 1w EMA trend filter
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
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
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
        
        # Trend filter: only long in uptrend, only short in downtrend
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals with volume confirmation and trend filter
        breakout_long = volume_spike and uptrend and close[i] >= donchian_high[i]
        breakout_short = volume_spike and downtrend and close[i] <= donchian_low[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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