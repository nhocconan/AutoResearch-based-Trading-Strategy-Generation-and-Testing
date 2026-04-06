#!/usr/bin/env python3
"""
Experiment #12254: 1h Momentum with 4h Trend and Volume Confirmation
Hypothesis: 1h momentum captures short-term swings, filtered by 4h trend direction and volume confirmation.
Uses 4h/1d for signal direction (trend bias) and 1h only for entry timing. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
Works in bull markets by riding momentum and in bear by catching mean-reversion bounces against 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12254_1h_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum(close, period):
    """Calculate momentum"""
    mom = np.full_like(close, np.nan)
    mom[period:] = close[period:] - close[:-period]
    return mom

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI for trend
    rsi_4h = calculate_rsi(df_4h['close'].values, RSI_PERIOD)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-calculate hour for session filter
    hours = pd.DatetimeIndex(open_time).hour
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    rsi_1h = calculate_rsi(close, RSI_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 4h RSI not available
        if np.isnan(rsi_4h_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Momentum and RSI conditions
        mom_up = momentum[i] > 0 if not np.isnan(momentum[i]) else False
        mom_down = momentum[i] < 0 if not np.isnan(momentum[i]) else False
        rsi_oversold = rsi_1h[i] < RSI_OVERSOLD if not np.isnan(rsi_1h[i]) else False
        rsi_overbought = rsi_1h[i] > RSI_OVERBOUGHT if not np.isnan(rsi_1h[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (4h RSI)
        uptrend_4h = rsi_4h_aligned[i] > 50
        downtrend_4h = rsi_4h_aligned[i] < 50
        
        # Entry conditions
        # Long: momentum up + oversold RSI + volume + 4h uptrend
        long_entry = mom_up and rsi_oversold and volume_ok and uptrend_4h
        # Short: momentum down + overbought RSI + volume + 4h downtrend
        short_entry = mom_down and rsi_overbought and volume_ok and downtrend_4h
        
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
Experiment #12254: 1h Momentum with 4h Trend and Volume Confirmation
Hypothesis: 1h momentum captures short-term swings, filtered by 4h trend direction and volume confirmation.
Uses 4h/1d for signal direction (trend bias) and 1h only for entry timing. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
Works in bull markets by riding momentum and in bear by catching mean-reversion bounces against 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12254_1h_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum(close, period):
    """Calculate momentum"""
    mom = np.full_like(close, np.nan)
    mom[period:] = close[period:] - close[:-period]
    return mom

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI for trend
    rsi_4h = calculate_rsi(df_4h['close'].values, RSI_PERIOD)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-calculate hour for session filter
    hours = pd.DatetimeIndex(open_time).hour
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    rsi_1h = calculate_rsi(close, RSI_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 4h RSI not available
        if np.isnan(rsi_4h_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Momentum and RSI conditions
        mom_up = momentum[i] > 0 if not np.isnan(momentum[i]) else False
        mom_down = momentum[i] < 0 if not np.isnan(momentum[i]) else False
        rsi_oversold = rsi_1h[i] < RSI_OVERSOLD if not np.isnan(rsi_1h[i]) else False
        rsi_overbought = rsi_1h[i] > RSI_OVERBOUGHT if not np.isnan(rsi_1h[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (4h RSI)
        uptrend_4h = rsi_4h_aligned[i] > 50
        downtrend_4h = rsi_4h_aligned[i] < 50
        
        # Entry conditions
        # Long: momentum up + oversold RSI + volume + 4h uptrend
        long_entry = mom_up and rsi_oversold and volume_ok and uptrend_4h
        # Short: momentum down + overbought RSI + volume + 4h downtrend
        short_entry = mom_down and rsi_overbought and volume_ok and downtrend_4h
        
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