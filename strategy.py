#!/usr/bin/env python3
"""
exp_6727_6h_donchian20_1w_pivot_dir_v1
Hypothesis: 6h Donchian(20) breakout with 1-week pivot direction filter and volume confirmation.
In bull markets (price above weekly pivot): buy breakouts above upper Donchian with volume confirmation.
In bear markets (price below weekly pivot): sell breakdowns below lower Donchian with volume confirmation.
Weekly pivot provides robust regime filter that adapts to changing market structure.
Designed for 6h timeframe to capture medium-term swings with ~15-35 trades/year.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6727_6h_donchian20_1w_pivot_dir_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~3 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week pivot points (standard floor trader pivots)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = r3_1w + (high_1w - low_1w)
    s4_1w = s3_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed weeks only)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime from weekly pivot
        # Bull regime: price above weekly pivot
        # Bear regime: price below weekly pivot
        bull_regime = close[i] > pivot_1w_aligned[i]
        bear_regime = close[i] < pivot_1w_aligned[i]
        
        # Breakout signals with weekly pivot regime filter
        # In bull regime: buy breakouts above upper Donchian
        # In bear regime: sell breakdowns below lower Donchian
        long_breakout = bull_regime and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = bear_regime and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

#!/usr/bin/env python3
"""
exp_6727_6h_donchian20_1w_pivot_dir_v1
Hypothesis: 6h Donchian(20) breakout with 1-week pivot direction filter and volume confirmation.
In bull markets (price above weekly pivot): buy breakouts above upper Donchian with volume confirmation.
In bear markets (price below weekly pivot): sell breakdowns below lower Donchian with volume confirmation.
Weekly pivot provides robust regime filter that adapts to changing market structure.
Designed for 6h timeframe to capture medium-term swings with ~15-35 trades/year.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6727_6h_donchian20_1w_pivot_dir_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~3 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week pivot points (standard floor trader pivots)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = r3_1w + (high_1w - low_1w)
    s4_1w = s3_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed weeks only)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime from weekly pivot
        # Bull regime: price above weekly pivot
        # Bear regime: price below weekly pivot
        bull_regime = close[i] > pivot_1w_aligned[i]
        bear_regime = close[i] < pivot_1w_aligned[i]
        
        # Breakout signals with weekly pivot regime filter
        # In bull regime: buy breakouts above upper Donchian
        # In bear regime: sell breakdowns below lower Donchian
        long_breakout = bull_regime and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = bear_regime and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals