#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14051_6d_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Typical price
    pp = (high + low + close) / 3
    range_ = high - low
    
    # Camarilla levels
    r4 = pp + (range_ * 1.1 / 2)
    r3 = pp + (range_ * 1.1 / 4)
    s3 = pp - (range_ * 1.1 / 4)
    s4 = pp - (range_ * 1.1 / 2)
    
    return r4, r3, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using EMA of True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on 1d
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for Donchian breakout and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period) on 6h
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Calculate ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
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
        
        # Breakout conditions with Camarilla filter
        # Long: break above Donchian upper AND above R3 (bullish bias)
        # Short: break below Donchian lower AND below S3 (bearish bias)
        breakout_up = close[i] > donch_upper[i-1] and close[i] > r3_1d_aligned[i]
        breakout_down = close[i] < donch_lower[i-1] and close[i] < s3_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = vol_filter[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_down and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse signal
            if close[i] <= stop_price or (close[i] < donch_lower[i-1] and close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse signal
            if close[i] >= stop_price or (close[i] > donch_upper[i-1] and close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14051_6d_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Typical price
    pp = (high + low + close) / 3
    range_ = high - low
    
    # Camarilla levels
    r4 = pp + (range_ * 1.1 / 2)
    r3 = pp + (range_ * 1.1 / 4)
    s3 = pp - (range_ * 1.1 / 4)
    s4 = pp - (range_ * 1.1 / 2)
    
    return r4, r3, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using EMA of True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on 1d
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for Donchian breakout and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period) on 6h
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Calculate ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
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
        
        # Breakout conditions with Camarilla filter
        # Long: break above Donchian upper AND above R3 (bullish bias)
        # Short: break below Donchian lower AND below S3 (bearish bias)
        breakout_up = close[i] > donch_upper[i-1] and close[i] > r3_1d_aligned[i]
        breakout_down = close[i] < donch_lower[i-1] and close[i] < s3_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = vol_filter[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_down and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse signal
            if close[i] <= stop_price or (close[i] < donch_lower[i-1] and close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse signal
            if close[i] >= stop_price or (close[i] > donch_upper[i-1] and close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals