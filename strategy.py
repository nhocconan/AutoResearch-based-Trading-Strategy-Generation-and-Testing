#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4-hour and 1-day trend filters
# Uses 4-hour Supertrend for trend direction and 1-day RSI for momentum confirmation
# Entry only during active market hours (08-20 UTC) to reduce noise and false signals
# Target: 80-150 total trades over 4 years (20-38/year) for 1h timeframe
# Works in bull markets by following 4h trend and in bear markets by fading overextended moves
# when 1-day RSI shows extreme levels with volume confirmation

name = "exp_13554_1h_momentum_4h_supertrend_1d_rsi_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 10
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]):
            supertrend[i] = np.nan
            direction[i] = direction[i-1]
            continue
            
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else lowerband[i]
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else upperband[i]
            
    return supertrend, direction

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.nanmean(gain[:period])
    avg_loss[period] = np.nanmean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Supertrend for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    supertrend_4h, supertrend_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    supertrend_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_direction_4h.astype(float))
    
    # Calculate 1d RSI for momentum confirmation
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, RSI_PERIOD)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SUPERTREND_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(supertrend_direction_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check session filter (08-20 UTC)
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: 4h Supertrend direction
        uptrend_4h = supertrend_direction_4h_aligned[i] == 1
        downtrend_4h = supertrend_direction_4h_aligned[i] == -1
        
        # Momentum filter: 1d RSI extremes
        rsi_overbought = rsi_1d_aligned[i] > RSI_OVERBOUGHT
        rsi_oversold = rsi_1d_aligned[i] < RSI_OVERSOLD
        
        # Entry signals
        if position == 0 and in_session and volume_ok:
            # Long: 4h uptrend + 1-day RSI oversold (mean reversion in uptrend)
            if uptrend_4h and rsi_oversold:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: 4h downtrend + 1-day RSI overbought (mean reversion in downtrend)
            elif downtrend_4h and rsi_overbought:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position != 0:
            signals[i] = position * SIGNAL_SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4-hour Supertrend and 1-day RSI filters
# Uses 4h Supertrend for trend direction and 1d RSI for mean reversion entries
# Entry only during active market hours (08-20 UTC) to reduce noise
# Target: 80-150 total trades over 4 years (20-38/year) for 1h timeframe
# Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends

name = "exp_13554_1h_momentum_4h_supertrend_1d_rsi_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 10
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]):
            supertrend[i] = np.nan
            direction[i] = direction[i-1]
            continue
            
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else lowerband[i]
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else upperband[i]
            
    return supertrend, direction

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.nanmean(gain[:period])
    avg_loss[period] = np.nanmean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Supertrend for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    supertrend_4h, supertrend_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    supertrend_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_direction_4h.astype(float))
    
    # Calculate 1d RSI for momentum confirmation
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, RSI_PERIOD)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SUPERTREND_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(supertrend_direction_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check session filter (08-20 UTC)
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: 4h Supertrend direction
        uptrend_4h = supertrend_direction_4h_aligned[i] == 1
        downtrend_4h = supertrend_direction_4h_aligned[i] == -1
        
        # Momentum filter: 1d RSI extremes
        rsi_overbought = rsi_1d_aligned[i] > RSI_OVERBOUGHT
        rsi_oversold = rsi_1d_aligned[i] < RSI_OVERSOLD
        
        # Entry signals
        if position == 0 and in_session and volume_ok:
            # Long: 4h uptrend + 1-day RSI oversold (mean reversion in uptrend)
            if uptrend_4h and rsi_oversold:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: 4h downtrend + 1-day RSI overbought (mean reversion in downtrend)
            elif downtrend_4h and rsi_overbought:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position != 0:
            signals[i] = position * SIGNAL_SIZE
        else:
            signals[i] = 0.0
    
    return signals