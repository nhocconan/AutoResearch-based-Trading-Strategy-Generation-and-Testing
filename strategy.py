# %%
#!/usr/bin/env python3
# 4H_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (4h) for trend direction, RSI(14) for momentum, and Choppiness Index (4h) for regime filter.
# Enters long when KAMA is rising (bullish), RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
# Enters short when KAMA is falling (bearish), RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
# Exits when RSI returns to neutral (40-60) or trend changes.
# Uses 4h timeframe with position size 0.25 to limit trades (~25-40/year) and minimize fee drag.
# Works in both bull/bear markets by mean-reverting in ranges and following KAMA trend.

name = "4H_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - momentum oscillator
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when not enough data
    
    # Choppiness Index (14) - regime filter
    atr1 = np.maximum(high - low, 
                     np.maximum(abs(high - close_s.shift(1)), 
                                abs(low - close_s.shift(1))))
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA slope
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Market regime: choppy (range-bound) vs trending
        is_choppy = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long entry: oversold in ranging market with bullish KAMA
            if (rsi[i] < 30 and 
                kama_rising and 
                is_choppy):
                signals[i] = 0.25
                position = 1
            # Short entry: overbought in ranging market with bearish KAMA
            elif (rsi[i] > 70 and 
                  kama_falling and 
                  is_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if (rsi[i] > 40 or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if (rsi[i] < 60 or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%
#!/usr/bin/env python3
# 4H_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (4h) for trend direction, RSI(14) for momentum, and Choppiness Index (4h) for regime filter.
# Enters long when KAMA is rising (bullish), RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
# Enters short when KAMA is falling (bearish), RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
# Exits when RSI returns to neutral (40-60) or trend changes.
# Uses 4h timeframe with position size 0.25 to limit trades (~25-40/year) and minimize fee drag.
# Works in both bull/bear markets by mean-reverting in ranges and following KAMA trend.

name = "4H_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - momentum oscillator
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when not enough data
    
    # Choppiness Index (14) - regime filter
    atr1 = np.maximum(high - low, 
                     np.maximum(abs(high - close_s.shift(1)), 
                                abs(low - close_s.shift(1))))
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA slope
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Market regime: choppy (range-bound) vs trending
        is_choppy = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long entry: oversold in ranging market with bullish KAMA
            if (rsi[i] < 30 and 
                kama_rising and 
                is_choppy):
                signals[i] = 0.25
                position = 1
            # Short entry: overbought in ranging market with bearish KAMA
            elif (rsi[i] > 70 and 
                  kama_falling and 
                  is_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if (rsi[i] > 40 or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if (rsi[i] < 60 or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%
#!/usr/bin/env python3
# 4H_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (4h) for trend direction, RSI(14) for momentum, and Choppiness Index (4h) for regime filter.
# Enters long when KAMA is rising (bullish), RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
# Enters short when KAMA is falling (bearish), RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
# Exits when RSI returns to neutral (40-60) or trend changes.
# Uses 4h timeframe with position size 0.25 to limit trades (~25-40/year) and minimize fee drag.
# Works in both bull/bear markets by mean-reverting in ranges and following KAMA trend.

name = "4H_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - momentum oscillator
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when not enough data
    
    # Choppiness Index (14) - regime filter
    atr1 = np.maximum(high - low, 
                     np.maximum(abs(high - close_s.shift(1)), 
                                abs(low - close_s.shift(1))))
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA slope
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Market regime: choppy (range-bound) vs trending
        is_choppy = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long entry: oversold in ranging market with bullish KAMA
            if (rsi[i] < 30 and 
                kama_rising and 
                is_choppy):
                signals[i] = 0.25
                position = 1
            # Short entry: overbought in ranging market with bearish KAMA
            elif (rsi[i] > 70 and 
                  kama_falling and 
                  is_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if (rsi[i] > 40 or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if (rsi[i] < 60 or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%
#!/usr/bin/env python3
# 4H_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (4h) for trend direction, RSI(14) for momentum, and Choppiness Index (4h) for regime filter.
# Enters long when KAMA is rising (bullish), RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
# Enters short when KAMA is falling (bearish), RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
# Exits when RSI returns to neutral (40-60) or trend changes.
# Uses 4h timeframe with position size 0.25 to limit trades (~25-40/year) and minimize fee drag.
# Works in both bull/bear markets by mean-reverting in ranges and following KAMA trend.

name = "4H_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - momentum oscillator
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when not enough data
    
    # Choppiness Index (14) - regime filter
    atr1 = np.maximum(high - low, 
                     np.maximum(abs(high - close_s.shift(1)), 
                                abs(low - close_s.shift(1))))
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA slope
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Market regime: choppy (range-bound) vs trending
        is_choppy = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long entry: oversold in ranging market with bullish KAMA
            if (rsi[i] < 30 and 
                kama_rising and 
                is_choppy):
                signals[i] = 0.25
                position = 1
            # Short entry: overbought in ranging market with bearish KAMA
            elif (rsi[i] > 70 and 
                  kama_falling and 
                  is_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if (rsi[i] > 40 or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if (rsi[i] < 60 or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%
#!/usr/bin/env python3
# 4H_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (4h) for trend direction, RSI(14) for momentum, and Choppiness Index (4h) for regime filter.
# Enters long when KAMA is rising (bullish), RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
# Enters short when KAMA is falling (bearish), RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
# Exits when RSI returns to neutral (40-60) or trend changes.
# Uses 4h timeframe with position size 0.25 to limit trades (~25-40/year) and minimize fee drag.
# Works in both bull/bear markets by mean-reverting in ranges and following KAMA trend.

name = "4H_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - momentum oscillator
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when not enough data
    
    # Choppiness Index (14) - regime filter
    atr1 = np.maximum(high - low, 
                     np.maximum(abs(high - close_s.shift(1)), 
                                abs(low - close_s.shift(1))))
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA slope
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Market regime: choppy (range-bound) vs trending
        is_choppy = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long entry: oversold in ranging market with bullish KAMA
            if (rsi[i] < 30 and 
                kama_rising and 
                is_choppy):
                signals[i] = 0.25
                position = 1
            # Short entry: overbought in ranging market with bearish KAMA
            elif (rsi[i] > 70 and 
                  kama_falling and 
                  is_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if (rsi[i] > 40 or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if (rsi[i] < 60 or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%
#!/usr/bin/env python3
# 4H_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (4h) for trend direction, RSI(14) for momentum, and Choppiness Index (4h) for regime filter.
# Enters long when KAMA is rising (bullish), RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
# Enters short when KAMA is falling (bearish), RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
# Exits when RSI returns to neutral (40-60) or trend changes.
# Uses 4h timeframe with position size 0.25 to limit trades (~25-40/year) and minimize fee drag.
# Works in both bull/bear markets by mean-reverting in ranges and following KAMA trend.

name = "4H_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - momentum oscillator
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when not enough data
    
    # Choppiness Index (14) - regime filter
    atr1 = np.maximum(high - low, 
                     np.maximum(abs(high - close_s.shift(1)), 
                                abs(low - close_s.shift(1))))
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA slope
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Market regime: choppy (range-bound) vs trending
        is_choppy = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long entry: oversold in ranging market with bullish KAMA
            if (rsi[i] < 30 and 
                kama_rising and 
                is_choppy):
                signals[i] = 0.25
                position = 1
            # Short entry: overbought in ranging market with bearish KAMA
            elif (rsi[i] > 70 and 
                  kama_falling and 
                  is_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if (rsi[i] > 40 or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if (rsi[i] < 60 or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%
#!/usr/bin/env python3
# 4H_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (4h) for trend direction, RSI(14) for momentum, and Choppiness Index (4h) for regime filter.
# Enters long when KAMA is rising (bullish), RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
# Enters short when KAMA is falling (bearish), RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
# Exits when RSI returns to neutral (40-60) or trend changes.
# Uses 4h timeframe with position size 0.25 to limit trades (~25-40/year) and minimize fee drag.
# Works in both bull/bear markets by mean-reverting in ranges and following KAMA trend.

name = "4H_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices