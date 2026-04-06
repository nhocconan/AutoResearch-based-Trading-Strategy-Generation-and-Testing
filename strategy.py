#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14062_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(arr, period):
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA (50-period)
    ema_50 = calculate_ema(close_1d, 50)
    
    # Calculate 1d volume SMA (20-period) for volume confirmation
    volume_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20)
    
    # 12h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h volume SMA (20-period) for volume confirmation
    volume_sma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA, 14 for ATR)
    start = max(20, 50, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or \
           np.isnan(volume_sma_12h[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Breakout conditions with volume confirmation
        # Long: price breaks above Donchian high + volume > 1.5x avg + price > EMA50
        long_breakout = close[i] > high_max_20_aligned[i] and \
                        volume[i] > 1.5 * volume_sma_12h[i] and \
                        close[i] > ema_50_aligned[i]
        
        # Short: price breaks below Donchian low + volume > 1.5x avg + price < EMA50
        short_breakout = close[i] < low_min_20_aligned[i] and \
                         volume[i] > 1.5 * volume_sma_12h[i] and \
                         close[i] < ema_50_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal breakout
            if close[i] <= stop_price or short_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reversal breakout
            if close[i] >= stop_price or long_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14062_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(arr, period):
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA (50-period)
    ema_50 = calculate_ema(close_1d, 50)
    
    # Calculate 1d volume SMA (20-period) for volume confirmation
    volume_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20)
    
    # 12h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h volume SMA (20-period) for volume confirmation
    volume_sma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA, 14 for ATR)
    start = max(20, 50, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or \
           np.isnan(volume_sma_12h[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Breakout conditions with volume confirmation
        # Long: price breaks above Donchian high + volume > 1.5x avg + price > EMA50
        long_breakout = close[i] > high_max_20_aligned[i] and \
                        volume[i] > 1.5 * volume_sma_12h[i] and \
                        close[i] > ema_50_aligned[i]
        
        # Short: price breaks below Donchian low + volume > 1.5x avg + price < EMA50
        short_breakout = close[i] < low_min_20_aligned[i] and \
                         volume[i] > 1.5 * volume_sma_12h[i] and \
                         close[i] < ema_50_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal breakout
            if close[i] <= stop_price or short_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reversal breakout
            if close[i] >= stop_price or long_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals