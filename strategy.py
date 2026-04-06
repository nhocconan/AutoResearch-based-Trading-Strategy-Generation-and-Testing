#!/usr/bin/env python3
"""
4h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Breakouts from Donchian channels (20-period) align with daily trend and volume
provide high-probability entries in both bull and bear markets. Uses 1d EMA(50) for trend
filter to avoid counter-trend trades. Volume filter ensures participation. Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14266_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) from previous bar
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = position * 0.30
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
        
        # Donchian breakout signals with 1d trend filter and volume
        # Long: close > Donchian high + price > 1d EMA + volume
        # Short: close < Donchian low + price < 1d EMA + volume
        long_signal = close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]
        short_signal = close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price returns to Donchian mean (exit at midpoint)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= stop_price or close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short on stop or when price returns to Donchian mean
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= stop_price or close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

</think>
#!/usr/bin/env python3
"""
4h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Breakouts from Donchian channels (20-period) align with daily trend and volume
provide high-probability entries in both bull and bear markets. Uses 1d EMA(50) for trend
filter to avoid counter-trend trades. Volume filter ensures participation. Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14266_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) from previous bar
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = position * 0.30
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
        
        # Donchian breakout signals with 1d trend filter and volume
        # Long: close > Donchian high + price > 1d EMA + volume
        # Short: close < Donchian low + price < 1d EMA + volume
        long_signal = close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]
        short_signal = close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price returns to Donchian mean (exit at midpoint)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= stop_price or close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short on stop or when price returns to Donchian mean
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= stop_price or close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

</think>
#!/usr/bin/env python3
"""
4h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Breakouts from Donchian channels (20-period) align with daily trend and volume
provide high-probability entries in both bull and bear markets. Uses 1d EMA(50) for trend
filter to avoid counter-trend trades. Volume filter ensures participation. Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14266_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) from previous bar
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = position * 0.30
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
        
        # Donchian breakout signals with 1d trend filter and volume
        # Long: close > Donchian high + price > 1d EMA + volume
        # Short: close < Donchian low + price < 1d EMA + volume
        long_signal = close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]
        short_signal = close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price returns to Donchian mean (exit at midpoint)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= stop_price or close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short on stop or when price returns to Donchian mean
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= stop_price or close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

</think>
#!/usr/bin/env python3
"""
4h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Breakouts from Donchian channels (20-period) align with daily trend and volume
provide high-probability entries in both bull and bear markets. Uses 1d EMA(50) for trend
filter to avoid counter-trend trades. Volume filter ensures participation. Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14266_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) from previous bar
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = position * 0.30
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
        
        # Donchian breakout signals with 1d trend filter and volume
        # Long: close > Donchian high + price > 1d EMA + volume
        # Short: close < Donchian low + price < 1d EMA + volume
        long_signal = close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]
        short_signal = close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price returns to Donchian mean (exit at midpoint)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= stop_price or close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short on stop or when price returns to Donchian mean
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= stop_price or close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

</think>
#!/usr/bin/env python3
"""
4h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Breakouts from Donchian channels (20-period) align with daily trend and volume
provide high-probability entries in both bull and bear markets. Uses 1d EMA(50) for trend
filter to avoid counter-trend trades. Volume filter ensures participation. Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14266_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) from previous bar
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = position * 0.30
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
        
        # Donchian breakout signals with 1d trend filter and volume
        # Long: close > Donchian high + price > 1d EMA + volume
        # Short: close < Donchian low + price < 1d EMA + volume
        long_signal = close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]
        short_signal = close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price returns to Donchian mean (exit at midpoint)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= stop_price or close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short on stop or when price returns to Donchian mean
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= stop_price or close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

#!/usr/bin/env python3
"""
4h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Breakouts from Donchian channels (20-period) align with daily trend and volume
provide high-probability entries in both bull and bear markets. Uses 1d EMA(50) for trend
filter to avoid counter-trend trades. Volume filter ensures participation. Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14266_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) from previous bar
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series