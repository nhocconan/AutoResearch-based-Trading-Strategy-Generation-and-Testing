#!/usr/bin/env python3
"""
1h trend following with 4h/1d trend filter and volume confirmation.
Long when: price > 4h EMA(20), price > 1d EMA(50), and volume > 1.5x average
Short when: price < 4h EMA(20), price < 1d EMA(50), and volume > 1.5x average
Exit: stop loss (2*ATR) or reversal signal
Position size: 0.20
Target: 60-150 total trades over 4 years (15-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14234_1h_ema_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF EMAs
    ema_4h = calculate_ema(df_4h['close'].values, 20)
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    
    # Align HTF EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 50, 20, 14) + 1  # EMA periods and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or \
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
        
        # Trend following signals with HTF EMA filter and volume
        # Long: price > 4h EMA AND price > 1d EMA AND volume
        # Short: price < 4h EMA AND price < 1d EMA AND volume
        long_condition = (close[i] > ema_4h_aligned[i]) and (close[i] > ema_1d_aligned[i]) and vol_filter[i]
        short_condition = (close[i] < ema_4h_aligned[i]) and (close[i] < ema_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal (price < 4h EMA)
            if close[i] <= stop_price or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on stop or reversal (price > 4h EMA)
            if close[i] >= stop_price or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

</think>

#!/usr/bin/env python3
"""
1h trend following with 4h/1d trend filter and volume confirmation.
Long when: price > 4h EMA(20), price > 1d EMA(50), and volume > 1.5x average
Short when: price < 4h EMA(20), price < 1d EMA(50), and volume > 1.5x average
Exit: stop loss (2*ATR) or reversal signal
Position size: 0.20
Target: 60-150 total trades over 4 years (15-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14234_1h_ema_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF EMAs
    ema_4h = calculate_ema(df_4h['close'].values, 20)
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    
    # Align HTF EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 50, 20, 14) + 1  # EMA periods and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or \
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
        
        # Trend following signals with HTF EMA filter and volume
        # Long: price > 4h EMA AND price > 1d EMA AND volume
        # Short: price < 4h EMA AND price < 1d EMA AND volume
        long_condition = (close[i] > ema_4h_aligned[i]) and (close[i] > ema_1d_aligned[i]) and vol_filter[i]
        short_condition = (close[i] < ema_4h_aligned[i]) and (close[i] < ema_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal (price < 4h EMA)
            if close[i] <= stop_price or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on stop or reversal (price > 4h EMA)
            if close[i] >= stop_price or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

</think>