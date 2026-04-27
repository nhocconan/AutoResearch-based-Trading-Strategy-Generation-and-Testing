#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (excluding current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        weekly_trend_up = ema_50_1w_aligned[i] > 0  # EMA value itself indicates trend
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter: only go long in uptrend, short in downtrend
        weekly_uptrend = ema_50_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False
        weekly_downtrend = ema_50_1w_aligned[i] < close_1w[-1] if len(close_1w) > 0 else False
        
        # More robust weekly trend using EMA slope
        if i >= 51:
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > highest_high[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < lowest_low[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < lowest_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > highest_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (excluding current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        weekly_trend_up = ema_50_1w_aligned[i] > 0  # EMA value itself indicates trend
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter: only go long in uptrend, short in downtrend
        weekly_uptrend = ema_50_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False
        weekly_downtrend = ema_50_1w_aligned[i] < close_1w[-1] if len(close_1w) > 0 else False
        
        # More robust weekly trend using EMA slope
        if i >= 51:
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > highest_high[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < lowest_low[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < lowest_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > highest_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (excluding current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter using EMA slope (more robust)
        if i >= 51:
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > highest_high[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < lowest_low[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < lowest_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > highest_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (excluding current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter using EMA slope (more robust)
        if i >= 51:
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > highest_high[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < lowest_low[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < lowest_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > highest_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (excluding current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter using EMA slope (more robust)
        if i >= 51:
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > highest_high[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < lowest_low[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < lowest_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > highest_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (excluding current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter using EMA slope (more robust)
        if i >= 51:
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > highest_high[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < lowest_low[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < lowest_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > highest_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (excluding current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter using EMA slope (more robust)
        if i >= 51:
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > highest_high[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < lowest_low[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < lowest_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > highest_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, works in bear via filtering out false breaks
# Weekly trend ensures we only trade with the dominant higher timeframe direction
# Volume confirms institutional participation in the breakout