#!/usr/bin/env python3
"""
1h Donchian breakout with 4h trend filter and volume confirmation.
Hypothesis: 1h price breaks above/below 4h Donchian channels capture trend continuation.
Volume confirms breakout strength. 4h trend filter ensures alignment with higher timeframe trend.
Trades only during active London/NY session (08-20 UTC) to reduce noise.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14314_1h_donchian_4htrend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_high_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_low_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h timeframe (already shifted by 1 for completed bars)
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ma)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian)
    start = 20 + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal if price crosses opposite Donchian band
        if position == 1:  # long position
            if close[i] < donchian_low_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > donchian_high_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 1h price breaks 4h Donchian with volume and session
            long_setup = (close[i] > donchian_high_4h_aligned[i-1]) and vol_confirm[i] and session_mask[i]
            short_setup = (close[i] < donchian_low_4h_aligned[i-1]) and vol_confirm[i] and session_mask[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
1h Donchian breakout with 4h trend filter and volume confirmation.
Hypothesis: 1h price breaks above/below 4h Donchian channels capture trend continuation.
Volume confirms breakout strength. 4h trend filter ensures alignment with higher timeframe trend.
Trades only during active London/NY session (08-20 UTC) to reduce noise.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14314_1h_donchian_4htrend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_high_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_low_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h timeframe (already shifted by 1 for completed bars)
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ma)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian)
    start = 20 + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal if price crosses opposite Donchian band
        if position == 1:  # long position
            if close[i] < donchian_low_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > donchian_high_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 1h price breaks 4h Donchian with volume and session
            long_setup = (close[i] > donchian_high_4h_aligned[i-1]) and vol_confirm[i] and session_mask[i]
            short_setup = (close[i] < donchian_low_4h_aligned[i-1]) and vol_confirm[i] and session_mask[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals