#!/usr/bin/env python3
"""
4h Donchian breakout with 1d trend filter and volume confirmation.
Hypothesis: Donchian breakouts capture trend continuation. The 1d EMA filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws. Volume confirmation filters false breakouts. This should work in both bull and bear markets by following the dominant trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14303_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA)
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Look for entries: Donchian breakout with trend and volume confirmation
        # Long when price breaks above Donchian high in uptrend with volume
        # Short when price breaks below Donchian low in downtrend with volume
        long_setup = (close[i] > donchian_high[i-1]) and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
        short_setup = (close[i] < donchian_low[i-1]) and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
        
        if long_setup and position <= 0:
            signals[i] = 0.25
            position = 1
            entry_price = close[i]
        elif short_setup and position >= 0:
            signals[i] = -0.25
            position = -1
            entry_price = close[i]
        else:
            # Hold current position
            signals[i] = position * 0.25 if position != 0 else 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
4h Donchian breakout with 1d trend filter and volume confirmation.
Hypothesis: Donchian breakouts capture trend continuation. The 1d EMA filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws. Volume confirmation filters false breakouts. This should work in both bull and bear markets by following the dominant trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14303_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA)
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Look for entries: Donchian breakout with trend and volume confirmation
        # Long when price breaks above Donchian high in uptrend with volume
        # Short when price breaks below Donchian low in downtrend with volume
        long_setup = (close[i] > donchian_high[i-1]) and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
        short_setup = (close[i] < donchian_low[i-1]) and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
        
        if long_setup and position <= 0:
            signals[i] = 0.25
            position = 1
            entry_price = close[i]
        elif short_setup and position >= 0:
            signals[i] = -0.25
            position = -1
            entry_price = close[i]
        else:
            # Hold current position
            signals[i] = position * 0.25 if position != 0 else 0.0
    
    return signals