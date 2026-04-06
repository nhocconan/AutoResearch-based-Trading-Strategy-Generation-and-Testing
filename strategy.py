#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14067_6d_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper=period high, lower=period low"""
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return high_roll, low_roll

def calculate_pivot(high, low, close):
    """Calculate classic pivot points: P=(H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L)"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return p, r1, s1, r2, s2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pw_1w, r1_1w, s1_1w, r2_1w, s2_1w = calculate_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly pivot points to 6h timeframe
    pw_1w_aligned = align_htf_to_ltf(prices, df_1w, pw_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Load daily data for volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 6h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 6h (20-period)
    upper_6h, lower_6h = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume MA, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or \
           np.isnan(pw_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr[i]):
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
        
        # Volume filter: current volume > 20-day average volume
        vol_filter = volume[i] > vol_ma_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_6h[i]
        breakout_down = close[i] < lower_6h[i]
        
        # Weekly pivot direction: price above/below weekly pivot
        price_above_pivot = close[i] > pw_1w_aligned[i]
        price_below_pivot = close[i] < pw_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: Donchian breakout up + price above weekly pivot + volume filter
            if breakout_up and price_above_pivot and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: Donchian breakout down + price below weekly pivot + volume filter
            elif breakout_down and price_below_pivot and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or Donchian breakdown
            if close[i] <= stop_price or close[i] < lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or Donchian breakout up
            if close[i] >= stop_price or close[i] > upper_6h[i]:
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

name = "exp_14067_6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper=period high, lower=period low"""
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return high_roll, low_roll

def calculate_pivot(high, low, close):
    """Calculate classic pivot points: P=(H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L)"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return p, r1, s1, r2, s2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pw_1w, r1_1w, s1_1w, r2_1w, s2_1w = calculate_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly pivot points to 6h timeframe
    pw_1w_aligned = align_htf_to_ltf(prices, df_1w, pw_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Load daily data for volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 6h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 6h (20-period)
    upper_6h, lower_6h = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume MA, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or \
           np.isnan(pw_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr[i]):
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
        
        # Volume filter: current volume > 20-day average volume
        vol_filter = volume[i] > vol_ma_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_6h[i]
        breakout_down = close[i] < lower_6h[i]
        
        # Weekly pivot direction: price above/below weekly pivot
        price_above_pivot = close[i] > pw_1w_aligned[i]
        price_below_pivot = close[i] < pw_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: Donchian breakout up + price above weekly pivot + volume filter
            if breakout_up and price_above_pivot and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: Donchian breakout down + price below weekly pivot + volume filter
            elif breakout_down and price_below_pivot and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or Donchian breakdown
            if close[i] <= stop_price or close[i] < lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or Donchian breakout up
            if close[i] >= stop_price or close[i] > upper_6h[i]:
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

name = "exp_14067_6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper=period high, lower=period low"""
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return high_roll, low_roll

def calculate_pivot(high, low, close):
    """Calculate classic pivot points: P=(H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L)"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return p, r1, s1, r2, s2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pw_1w, r1_1w, s1_1w, r2_1w, s2_1w = calculate_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly pivot points to 6h timeframe
    pw_1w_aligned = align_htf_to_ltf(prices, df_1w, pw_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Load daily data for volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 6h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 6h (20-period)
    upper_6h, lower_6h = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume MA, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or \
           np.isnan(pw_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr[i]):
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
        
        # Volume filter: current volume > 20-day average volume
        vol_filter = volume[i] > vol_ma_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_6h[i]
        breakout_down = close[i] < lower_6h[i]
        
        # Weekly pivot direction: price above/below weekly pivot
        price_above_pivot = close[i] > pw_1w_aligned[i]
        price_below_pivot = close[i] < pw_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: Donchian breakout up + price above weekly pivot + volume filter
            if breakout_up and price_above_pivot and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: Donchian breakout down + price below weekly pivot + volume filter
            elif breakout_down and price_below_pivot and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or Donchian breakdown
            if close[i] <= stop_price or close[i] < lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or Donchian breakout up
            if close[i] >= stop_price or close[i] > upper_6h[i]:
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

name = "exp_14067_6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper=period high, lower=period low"""
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return high_roll, low_roll

def calculate_pivot(high, low, close):
    """Calculate classic pivot points: P=(H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L)"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return p, r1, s1, r2, s2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pw_1w, r1_1w, s1_1w, r2_1w, s2_1w = calculate_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly pivot points to 6h timeframe
    pw_1w_aligned = align_htf_to_ltf(prices, df_1w, pw_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Load daily data for volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 6h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 6h (20-period)
    upper_6h, lower_6h = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume MA, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or \
           np.isnan(pw_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr[i]):
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
        
        # Volume filter: current volume > 20-day average volume
        vol_filter = volume[i] > vol_ma_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_6h[i]
        breakout_down = close[i] < lower_6h[i]
        
        # Weekly pivot direction: price above/below weekly pivot
        price_above_pivot = close[i] > pw_1w_aligned[i]
        price_below_pivot = close[i] < pw_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: Donchian breakout up + price above weekly pivot + volume filter
            if breakout_up and price_above_pivot and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: Donchian breakout down + price below weekly pivot + volume filter
            elif breakout_down and price_below_pivot and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or Donchian breakdown
            if close[i] <= stop_price or close[i] < lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or Donchian breakout up
            if close[i] >= stop_price or close[i] > upper_6h[i]:
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

name = "exp_14067_6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper=period high, lower=period low"""
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return high_roll, low_roll

def calculate_pivot(high, low, close):
    """Calculate classic pivot points: P=(H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L)"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return p, r1, s1, r2, s2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pw_1w, r1_1w, s1_1w, r2_1w, s2_1w = calculate_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly pivot points to 6h timeframe
    pw_1w_aligned = align_htf_to_ltf(prices, df_1w, pw_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Load daily data for volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_ht