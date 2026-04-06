#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12853_4h_12h_camarilla_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = np.full_like(close_12h, np.nan)
    r1_12h = np.full_like(close_12h, np.nan)
    r2_12h = np.full_like(close_12h, np.nan)
    r3_12h = np.full_like(close_12h, np.nan)
    s1_12h = np.full_like(close_12h, np.nan)
    s2_12h = np.full_like(close_12h, np.nan)
    s3_12h = np.full_like(close_12h, np.nan)
    
    # Vectorized Camarilla calculation
    for i in range(len(close_12h)):
        p = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        r = high_12h[i] - low_12h[i]
        pivot_12h[i] = p
        r1_12h[i] = p + (r * 1.1 / 12)
        r2_12h[i] = p + (r * 1.1 / 6)
        r3_12h[i] = p + (r * 1.1 / 4)
        s1_12h[i] = p - (r * 1.1 / 12)
        s2_12h[i] = p - (r * 1.1 / 6)
        s3_12h[i] = p - (r * 1.1 / 4)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Calculate 4h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume moving average (24 periods = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 24  # volume MA period
    
    for i in range(start, n):
        # Check stop loss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Check if pivot levels are available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: Camarilla level touch with volume
        long_entry = volume_ok and close[i] <= s3_aligned[i] and close[i] > s2_aligned[i]
        short_entry = volume_ok and close[i] >= r3_aligned[i] and close[i] < r2_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (1.5 * atr[i])
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12853_4h_12h_camarilla_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = np.full_like(close_12h, np.nan)
    r1_12h = np.full_like(close_12h, np.nan)
    r2_12h = np.full_like(close_12h, np.nan)
    r3_12h = np.full_like(close_12h, np.nan)
    s1_12h = np.full_like(close_12h, np.nan)
    s2_12h = np.full_like(close_12h, np.nan)
    s3_12h = np.full_like(close_12h, np.nan)
    
    # Vectorized Camarilla calculation
    for i in range(len(close_12h)):
        p = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        r = high_12h[i] - low_12h[i]
        pivot_12h[i] = p
        r1_12h[i] = p + (r * 1.1 / 12)
        r2_12h[i] = p + (r * 1.1 / 6)
        r3_12h[i] = p + (r * 1.1 / 4)
        s1_12h[i] = p - (r * 1.1 / 12)
        s2_12h[i] = p - (r * 1.1 / 6)
        s3_12h[i] = p - (r * 1.1 / 4)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Calculate 4h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume moving average (24 periods = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 24  # volume MA period
    
    for i in range(start, n):
        # Check stop loss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Check if pivot levels are available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: Camarilla level touch with volume
        long_entry = volume_ok and close[i] <= s3_aligned[i] and close[i] > s2_aligned[i]
        short_entry = volume_ok and close[i] >= r3_aligned[i] and close[i] < r2_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (1.5 * atr[i])
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12853_4h_12h_camarilla_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = np.full_like(close_12h, np.nan)
    r1_12h = np.full_like(close_12h, np.nan)
    r2_12h = np.full_like(close_12h, np.nan)
    r3_12h = np.full_like(close_12h, np.nan)
    s1_12h = np.full_like(close_12h, np.nan)
    s2_12h = np.full_like(close_12h, np.nan)
    s3_12h = np.full_like(close_12h, np.nan)
    
    # Vectorized Camarilla calculation
    for i in range(len(close_12h)):
        p = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        r = high_12h[i] - low_12h[i]
        pivot_12h[i] = p
        r1_12h[i] = p + (r * 1.1 / 12)
        r2_12h[i] = p + (r * 1.1 / 6)
        r3_12h[i] = p + (r * 1.1 / 4)
        s1_12h[i] = p - (r * 1.1 / 12)
        s2_12h[i] = p - (r * 1.1 / 6)
        s3_12h[i] = p - (r * 1.1 / 4)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Calculate 4h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume moving average (24 periods = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 24  # volume MA period
    
    for i in range(start, n):
        # Check stop loss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Check if pivot levels are available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: Camarilla level touch with volume
        long_entry = volume_ok and close[i] <= s3_aligned[i] and close[i] > s2_aligned[i]
        short_entry = volume_ok and close[i] >= r3_aligned[i] and close[i] < r2_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (1.5 * atr[i])
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12853_4h_12h_camarilla_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = np.full_like(close_12h, np.nan)
    r1_12h = np.full_like(close_12h, np.nan)
    r2_12h = np.full_like(close_12h, np.nan)
    r3_12h = np.full_like(close_12h, np.nan)
    s1_12h = np.full_like(close_12h, np.nan)
    s2_12h = np.full_like(close_12h, np.nan)
    s3_12h = np.full_like(close_12h, np.nan)
    
    # Vectorized Camarilla calculation
    for i in range(len(close_12h)):
        p = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        r = high_12h[i] - low_12h[i]
        pivot_12h[i] = p
        r1_12h[i] = p + (r * 1.1 / 12)
        r2_12h[i] = p + (r * 1.1 / 6)
        r3_12h[i] = p + (r * 1.1 / 4)
        s1_12h[i] = p - (r * 1.1 / 12)
        s2_12h[i] = p - (r * 1.1 / 6)
        s3_12h[i] = p - (r * 1.1 / 4)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Calculate 4h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume moving average (24 periods = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 24  # volume MA period
    
    for i in range(start, n):
        # Check stop loss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Check if pivot levels are available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: Camarilla level touch with volume
        long_entry = volume_ok and close[i] <= s3_aligned[i] and close[i] > s2_aligned[i]
        short_entry = volume_ok and close[i] >= r3_aligned[i] and close[i] < r2_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (1.5 * atr[i])
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12853_4h_12h_camarilla_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = np.full_like(close_12h, np.nan)
    r1_12h = np.full_like(close_12h, np.nan)
    r2_12h = np.full_like(close_12h, np.nan)
    r3_12h = np.full_like(close_12h, np.nan)
    s1_12h = np.full_like(close_12h, np.nan)
    s2_12h = np.full_like(close_12h, np.nan)
    s3_12h = np.full_like(close_12h, np.nan)
    
    # Vectorized Camarilla calculation
    for i in range(len(close_12h)):
        p = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        r = high_12h[i] - low_12h[i]
        pivot_12h[i] = p
        r1_12h[i] = p + (r * 1.1 / 12)
        r2_12h[i] = p + (r * 1.1 / 6)
        r3_12h[i] = p + (r * 1.1 / 4)
        s1_12h[i] = p - (r * 1.1 / 12)
        s2_12h[i] = p - (r * 1.1 / 6)
        s3_12h[i] = p - (r * 1.1 / 4)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Calculate 4h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume moving average (24 periods = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 24  # volume MA period
    
    for i in range(start, n):
        # Check stop loss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Check if pivot levels are available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: Camarilla level touch with volume
        long_entry = volume_ok and close[i] <= s3_aligned[i] and close[i] > s2_aligned[i]
        short_entry = volume_ok and close[i] >= r3_aligned[i] and close[i] < r2_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (1.5 * atr[i])
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12853_4h_12h_camarilla_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = np.full_like(close_12h, np.nan)
    r1_12h = np.full_like(close_12h, np.nan)
    r2_12h = np.full_like(close_12h, np.nan)
    r3_12h = np.full_like(close_12h, np.nan)
    s1_12h = np.full_like(close_12h, np.nan)
    s2_12h = np.full_like(close_12h, np.nan)
    s3_12h = np.full_like(close_12h, np.nan)
    
    # Vectorized Camarilla calculation
    for i in range(len(close_12h)):
        p = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        r = high_12h[i] - low_12h[i]
        pivot_12h[i] = p
        r1_12h[i] = p + (r * 1.1 / 12)
        r2_12h[i] = p + (r * 1.1 / 6)
        r3_12h[i] = p + (r * 1.1 / 4)
        s1_12h[i] = p - (r * 1.1 / 12)
        s2_12h[i] = p - (r * 1.1 / 6)
        s3_12h[i] = p - (r * 1.1 / 4)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices