#!/usr/bin/env python3
"""
12h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Donchian(20) breakouts on 12h timeframe, filtered by 1d EMA(50) trend and volume spikes, capture medium-term trends while avoiding whipsaws. Volume confirmation ensures breakout strength, and EMA filter ensures alignment with daily trend. This should work in both bull and bear markets by capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14292_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 12h timeframe (shifted by 1 day for completed bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to EMA or opposite Donchian level
        if position == 1:  # long position
            if close[i] <= ema_1d_aligned[i] or close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_1d_aligned[i] or close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Donchian breakouts with trend and volume confirmation
            # Long when price breaks above Donchian high, price > EMA (uptrend) with volume confirmation
            # Short when price breaks below Donchian low, price < EMA (downtrend) with volume confirmation
            long_setup = (close[i] > donchian_high[i]) and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i]) and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
            
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
12h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Donchian(20) breakouts on 12h timeframe, filtered by 1d EMA(50) trend and volume spikes, capture medium-term trends while avoiding whipsaws. Volume confirmation ensures breakout strength, and EMA filter ensures alignment with daily trend. This should work in both bull and bear markets by capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14292_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 12h timeframe (shifted by 1 day for completed bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to EMA or opposite Donchian level
        if position == 1:  # long position
            if close[i] <= ema_1d_aligned[i] or close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_1d_aligned[i] or close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Donchian breakouts with trend and volume confirmation
            # Long when price breaks above Donchian high, price > EMA (uptrend) with volume confirmation
            # Short when price breaks below Donchian low, price < EMA (downtrend) with volume confirmation
            long_setup = (close[i] > donchian_high[i]) and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i]) and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
            
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