#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Hypothesis: Weekly pivot provides directional bias, Donchian breakout captures momentum,
and volume confirms strength. Works in bull (breakouts continue) and bear (breakdowns continue).
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14307_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly pivots to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 6h data
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
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, weekly data available)
    start = max(20, 50)  # Need enough for weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pivot_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or \
           np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: stoploss (2*ATR) or reversal at opposite pivot level
        if position == 1:  # long position
            if close[i] <= entry_price - 2.0 * atr[i] or close[i] >= r4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.0 * atr[i] or close[i] <= s4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly pivot bias and volume
            # Long when price breaks above Donchian high AND above weekly R4 (strong bullish bias)
            # Short when price breaks below Donchian low AND below weekly S4 (strong bearish bias)
            long_setup = (close[i] > donchian_high[i-1]) and (close[i] > r4_w_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i-1]) and (close[i] < s4_w_aligned[i]) and vol_confirm[i]
            
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
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Hypothesis: Weekly pivot provides directional bias, Donchian breakout captures momentum,
and volume confirms strength. Works in bull (breakouts continue) and bear (breakdowns continue).
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14307_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly pivots to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 6h data
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
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, weekly data available)
    start = max(20, 50)  # Need enough for weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pivot_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or \
           np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: stoploss (2*ATR) or reversal at opposite pivot level
        if position == 1:  # long position
            if close[i] <= entry_price - 2.0 * atr[i] or close[i] >= r4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.0 * atr[i] or close[i] <= s4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly pivot bias and volume
            # Long when price breaks above Donchian high AND above weekly R4 (strong bullish bias)
            # Short when price breaks below Donchian low AND below weekly S4 (strong bearish bias)
            long_setup = (close[i] > donchian_high[i-1]) and (close[i] > r4_w_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i-1]) and (close[i] < s4_w_aligned[i]) and vol_confirm[i]
            
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
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Hypothesis: Weekly pivot provides directional bias, Donchian breakout captures momentum,
and volume confirms strength. Works in bull (breakouts continue) and bear (breakdowns continue).
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14307_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly pivots to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 6h data
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
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, weekly data available)
    start = max(20, 50)  # Need enough for weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pivot_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or \
           np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: stoploss (2*ATR) or reversal at opposite pivot level
        if position == 1:  # long position
            if close[i] <= entry_price - 2.0 * atr[i] or close[i] >= r4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.0 * atr[i] or close[i] <= s4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly pivot bias and volume
            # Long when price breaks above Donchian high AND above weekly R4 (strong bullish bias)
            # Short when price breaks below Donchian low AND below weekly S4 (strong bearish bias)
            long_setup = (close[i] > donchian_high[i-1]) and (close[i] > r4_w_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i-1]) and (close[i] < s4_w_aligned[i]) and vol_confirm[i]
            
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
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Hypothesis: Weekly pivot provides directional bias, Donchian breakout captures momentum,
and volume confirms strength. Works in bull (breakouts continue) and bear (breakdowns continue).
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14307_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly pivots to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 6h data
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
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, weekly data available)
    start = max(20, 50)  # Need enough for weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pivot_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or \
           np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: stoploss (2*ATR) or reversal at opposite pivot level
        if position == 1:  # long position
            if close[i] <= entry_price - 2.0 * atr[i] or close[i] >= r4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.0 * atr[i] or close[i] <= s4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly pivot bias and volume
            # Long when price breaks above Donchian high AND above weekly R4 (strong bullish bias)
            # Short when price breaks below Donchian low AND below weekly S4 (strong bearish bias)
            long_setup = (close[i] > donchian_high[i-1]) and (close[i] > r4_w_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i-1]) and (close[i] < s4_w_aligned[i]) and vol_confirm[i]
            
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
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Hypothesis: Weekly pivot provides directional bias, Donchian breakout captures momentum,
and volume confirms strength. Works in bull (breakouts continue) and bear (breakdowns continue).
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14307_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly pivots to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 6h data
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
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, weekly data available)
    start = max(20, 50)  # Need enough for weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pivot_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or \
           np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: stoploss (2*ATR) or reversal at opposite pivot level
        if position == 1:  # long position
            if close[i] <= entry_price - 2.0 * atr[i] or close[i] >= r4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.0 * atr[i] or close[i] <= s4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly pivot bias and volume
            # Long when price breaks above Donchian high AND above weekly R4 (strong bullish bias)
            # Short when price breaks below Donchian low AND below weekly S4 (strong bearish bias)
            long_setup = (close[i] > donchian_high[i-1]) and (close[i] > r4_w_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i-1]) and (close[i] < s4_w_aligned[i]) and vol_confirm[i]
            
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
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Hypothesis: Weekly pivot provides directional bias, Donchian breakout captures momentum,
and volume confirms strength. Works in bull (breakouts continue) and bear (breakdowns continue).
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14307_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly pivots to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 6h data
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
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, weekly data available)
    start = max(20, 50)  # Need enough for weekly alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pivot_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or \
           np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: stoploss (2*ATR) or reversal at opposite pivot level
        if position == 1:  # long position
            if close[i] <= entry_price - 2.0 * atr[i] or close[i] >= r4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.0 * atr[i] or close[i] <= s4_w_aligned[i]:
                signals[i] = 0