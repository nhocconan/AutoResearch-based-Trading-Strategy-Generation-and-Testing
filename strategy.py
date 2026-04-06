#14045: 12h Donchian(20) breakout + 1d EMA filter + volume confirmation + ATR stop
# Hypothesis: Donchian breakouts capture trends; 1d EMA ensures trend alignment; volume confirms breakout strength.
# Works in bull (breakouts up) and bear (breakouts down) via directional filter.
# Target: 50-150 trades over 4 years (12-37/year) via tight 12h breakout conditions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14045_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
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
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA, 14 for ATR)
    start = max(20, 50, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i]) or np.isnan(atr[i]):
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
        
        # Breakout conditions
        bullish_breakout = close[i] > high_20[i-1] and volume[i] > vol_avg[i]
        bearish_breakout = close[i] < low_20[i-1] and volume[i] > vol_avg[i]
        
        # 1d EMA filter: align with higher timeframe trend
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        # Generate signals
        if position == 0:
            if bullish_breakout and price_above_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif bearish_breakout and price_below_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or breakout in opposite direction
            if close[i] <= stop_price or bearish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or breakout in opposite direction
            if close[i] >= stop_price or bullish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#14045: 12h Donchian(20) breakout + 1d EMA filter + volume confirmation + ATR stop
# Hypothesis: Donchian breakouts capture trends; 1d EMA ensures trend alignment; volume confirms breakout strength.
# Works in bull (breakouts up) and bear (breakouts down) via directional filter.
# Target: 50-150 trades over 4 years (12-37/year) via tight 12h breakout conditions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14045_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
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
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA, 14 for ATR)
    start = max(20, 50, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i]) or np.isnan(atr[i]):
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
        
        # Breakout conditions
        bullish_breakout = close[i] > high_20[i-1] and volume[i] > vol_avg[i]
        bearish_breakout = close[i] < low_20[i-1] and volume[i] > vol_avg[i]
        
        # 1d EMA filter: align with higher timeframe trend
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        # Generate signals
        if position == 0:
            if bullish_breakout and price_above_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif bearish_breakout and price_below_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or breakout in opposite direction
            if close[i] <= stop_price or bearish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or breakout in opposite direction
            if close[i] >= stop_price or bullish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals