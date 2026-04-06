#!/usr/bin/env python3
"""
1h momentum with 4h trend filter and volume confirmation
Hypothesis: In strong trends (4h EMA20), 1h momentum (RSI divergence + volume) captures retracement entries.
Works in bull: buy dips in uptrend. Works in bear: sell rallies in downtrend.
Volume confirms momentum, reducing false signals. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14254_1h_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period):
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(20)
    ema_4h = calculate_ema(close_4h, 20)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    rsi = calculate_rsi(close, 14)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Session filter: 8-20 UTC (inclusive)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start from warmup period (max of 14 for RSI, 20 for volume, 14 for ATR)
    start = max(14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if outside session or required data not available
        if not in_session[i] or np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Momentum signals with 4h EMA filter and volume
        # Long: RSI < 40 (oversold) + price > 4h EMA + volume
        # Short: RSI > 60 (overbought) + price < 4h EMA + volume
        long_signal = (rsi[i] < 40) and (close[i] > ema_4h_aligned[i]) and vol_filter[i]
        short_signal = (rsi[i] > 60) and (close[i] < ema_4h_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or RSI > 50 (momentum fade)
            if close[i] <= stop_price or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on stop or RSI < 50 (momentum fade)
            if close[i] >= stop_price or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h momentum with 4h trend filter and volume confirmation
Hypothesis: In strong trends (4h EMA20), 1h momentum (RSI divergence + volume) captures retracement entries.
Works in bull: buy dips in uptrend. Works in bear: sell rallies in downtrend.
Volume confirms momentum, reducing false signals. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14254_1h_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period):
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(20)
    ema_4h = calculate_ema(close_4h, 20)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    rsi = calculate_rsi(close, 14)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Session filter: 8-20 UTC (inclusive)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start from warmup period (max of 14 for RSI, 20 for volume, 14 for ATR)
    start = max(14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if outside session or required data not available
        if not in_session[i] or np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Momentum signals with 4h EMA filter and volume
        # Long: RSI < 40 (oversold) + price > 4h EMA + volume
        # Short: RSI > 60 (overbought) + price < 4h EMA + volume
        long_signal = (rsi[i] < 40) and (close[i] > ema_4h_aligned[i]) and vol_filter[i]
        short_signal = (rsi[i] > 60) and (close[i] < ema_4h_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or RSI > 50 (momentum fade)
            if close[i] <= stop_price or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on stop or RSI < 50 (momentum fade)
            if close[i] >= stop_price or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h momentum with 4h trend filter and volume confirmation
Hypothesis: In strong trends (4h EMA20), 1h momentum (RSI divergence + volume) captures retracement entries.
Works in bull: buy dips in uptrend. Works in bear: sell rallies in downtrend.
Volume confirms momentum, reducing false signals. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14254_1h_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period):
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(20)
    ema_4h = calculate_ema(close_4h, 20)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    rsi = calculate_rsi(close, 14)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Session filter: 8-20 UTC (inclusive)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start from warmup period (max of 14 for RSI, 20 for volume, 14 for ATR)
    start = max(14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if outside session or required data not available
        if not in_session[i] or np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Momentum signals with 4h EMA filter and volume
        # Long: RSI < 40 (oversold) + price > 4h EMA + volume
        # Short: RSI > 60 (overbought) + price < 4h EMA + volume
        long_signal = (rsi[i] < 40) and (close[i] > ema_4h_aligned[i]) and vol_filter[i]
        short_signal = (rsi[i] > 60) and (close[i] < ema_4h_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or RSI > 50 (momentum fade)
            if close[i] <= stop_price or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on stop or RSI < 50 (momentum fade)
            if close[i] >= stop_price or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

--- END OF FILE ---