#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d trend filter + volume confirmation.
Hypothesis: Donchian breakouts capture trend continuation, while 1d EMA filters for direction and volume confirms strength.
Works in bull markets via breakouts, in bear markets via short breakdowns, with volume filter reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14299_6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels with proper min_periods"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 6h timeframe (shifted by 1 day for completed bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian(20) channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 50 for EMA)
    start = max(20, 20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to EMA or opposite Donchian level
        if position == 1:  # long position
            if close[i] <= ema_1d_aligned[i] or close[i] >= donchian_upper[i] + (donchian_upper[i] - donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_1d_aligned[i] or close[i] <= donchian_lower[i] - (donchian_upper[i] - donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout of Donchian with trend and volume confirmation
            # Long when price breaks above upper Donchian in uptrend with volume
            # Short when price breaks below lower Donchian in downtrend with volume
            long_breakout = close[i] > donchian_upper[i]
            short_breakout = close[i] < donchian_lower[i]
            
            long_setup = long_breakout and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
            short_setup = short_breakout and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation.
Hypothesis: Donchian breakouts capture trend continuation, while 1d EMA filters for direction and volume confirms strength.
Works in bull markets via breakouts, in bear markets via short breakdowns, with volume filter reducing false signals.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14299_6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels with proper min_periods"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 6h timeframe (shifted by 1 day for completed bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian(20) channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 50 for EMA)
    start = max(20, 20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to EMA or opposite Donchian level
        if position == 1:  # long position
            if close[i] <= ema_1d_aligned[i] or close[i] >= donchian_upper[i] + (donchian_upper[i] - donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_1d_aligned[i] or close[i] <= donchian_lower[i] - (donchian_upper[i] - donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout of Donchian with trend and volume confirmation
            # Long when price breaks above upper Donchian in uptrend with volume
            # Short when price breaks below lower Donchian in downtrend with volume
            long_breakout = close[i] > donchian_upper[i]
            short_breakout = close[i] < donchian_lower[i]
            
            long_setup = long_breakout and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
            short_setup = short_breakout and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation.
Hypothesis: Donchian breakouts capture trend continuation, while 1d EMA filters for direction and volume confirms strength.
Works in bull markets via breakouts, in bear markets via short breakdowns, with volume filter reducing false signals.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14299_6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels with proper min_periods"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 6h timeframe (shifted by 1 day for completed bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian(20) channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 50 for EMA)
    start = max(20, 20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to EMA or opposite Donchian level
        if position == 1:  # long position
            if close[i] <= ema_1d_aligned[i] or close[i] >= donchian_upper[i] + (donchian_upper[i] - donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_1d_aligned[i] or close[i] <= donchian_lower[i] - (donchian_upper[i] - donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout of Donchian with trend and volume confirmation
            # Long when price breaks above upper Donchian in uptrend with volume
            # Short when price breaks below lower Donchian in downtrend with volume
            long_breakout = close[i] > donchian_upper[i]
            short_breakout = close[i] < donchian_lower[i]
            
            long_setup = long_breakout and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
            short_setup = short_breakout and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

---END---