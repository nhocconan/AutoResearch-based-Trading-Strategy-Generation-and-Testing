#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14116_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(arr, period):
    """Calculate EMA with proper min_periods"""
    s = pd.Series(arr)
    return s.ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1d data for Donchian channels and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels on 1d
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate EMA(50) on 1d
    ema_50 = calculate_ema(close_1d, 50)
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 12h data
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
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA, 20 for volume, 14 for ATR)
    start = max(20, 50, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or \
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
        
        # Donchian breakout with EMA trend filter and volume confirmation
        # Long: price breaks above Donchian high AND above EMA(50) + volume
        # Short: price breaks below Donchian low AND below EMA(50) + volume
        breakout_long = (close[i] > donchian_high_aligned[i-1]) and \
                        (close[i] > ema_50_aligned[i-1]) and \
                        vol_filter[i]
        breakout_short = (close[i] < donchian_low_aligned[i-1]) and \
                         (close[i] < ema_50_aligned[i-1]) and \
                         vol_filter[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price falls below Donchian low
            if close[i] <= stop_price or close[i] < donchian_low_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or when price rises above Donchian high
            if close[i] >= stop_price or close[i] > donchian_high_aligned[i-1]:
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

name = "exp_14116_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(arr, period):
    """Calculate EMA with proper min_periods"""
    s = pd.Series(arr)
    return s.ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1d data for Donchian channels and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels on 1d
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate EMA(50) on 1d
    ema_50 = calculate_ema(close_1d, 50)
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 12h data
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
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA, 20 for volume, 14 for ATR)
    start = max(20, 50, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or \
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
        
        # Donchian breakout with EMA trend filter and volume confirmation
        # Long: price breaks above Donchian high AND above EMA(50) + volume
        # Short: price breaks below Donchian low AND below EMA(50) + volume
        breakout_long = (close[i] > donchian_high_aligned[i-1]) and \
                        (close[i] > ema_50_aligned[i-1]) and \
                        vol_filter[i]
        breakout_short = (close[i] < donchian_low_aligned[i-1]) and \
                         (close[i] < ema_50_aligned[i-1]) and \
                         vol_filter[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price falls below Donchian low
            if close[i] <= stop_price or close[i] < donchian_low_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or when price rises above Donchian high
            if close[i] >= stop_price or close[i] > donchian_high_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>