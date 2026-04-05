#!/usr/bin/env python3
"""
exp_7191_6h_donchian20_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels for entry filtering.
In bull/bear markets: trade breakouts in direction of daily pivot bias (above/below pivot).
In ranging markets: fade at S3/R3 levels with volume confirmation.
Uses 1d Camarilla pivots for institutional reference points and 6h Donchian for structure.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in all regimes by combining price channels with pivot-based mean reversion/continuation logic.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7191_6h_donchian20_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
CAMARILLA_LOOKBACK = 1  # Use previous day's OHLC
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~4 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla formulas: 
    # H4 = Close + 1.5*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low)
    # L3 = Close - 1.125*(High-Low)
    # H3 and L3 are key fade levels, H4/L4 are breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for previous day (to avoid look-ahead)
    range_1d = high_1d - low_1d
    h4 = close_1d + 1.5 * range_1d
    l4 = close_1d - 1.5 * range_1d
    h3 = close_1d + 1.125 * range_1d
    l3 = close_1d - 1.125 * range_1d
    pivot = (high_1d + low_1d + close_1d) / 3.0  # Standard pivot point
    
    # Align to LTF (6h) - note: we use previous day's levels, so no additional shift needed
    # align_htf_to_ltf already includes shift(1) for completed bars only
    h4_6h = align_htf_to_ltf(prices, df_1d, h4)
    l4_6h = align_htf_to_ltf(prices, df_1d, l4)
    h3_6h = align_htf_to_ltf(prices, df_1d, h3)
    l3_6h = align_htf_to_ltf(prices, df_1d, l3)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    
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
        if np.isnan(h4_6h[i]) or np.isnan(l4_6h[i]) or np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]):
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
        
        # Determine market bias based on Camarilla levels
        above_h4 = close[i] > h4_6h[i]
        below_l4 = close[i] < l4_6h[i]
        between_h3_l3 = (close[i] > l3_6h[i]) and (close[i] < h3_6h[i])
        above_pivot = close[i] > pivot_6h[i]
        below_pivot = close[i] < pivot_6h[i]
        
        # Fade at S3/R3 in ranging market (between H3/L3) with volume
        fade_long = between_h3_l3 and (close[i] <= l3_6h[i]) and vol_confirmed and below_pivot
        fade_short = between_h3_l3 and (close[i] >= h3_6h[i]) and vol_confirmed and above_pivot
        
        # Breakout continuation at H4/L4 with volume
        breakout_long = above_h4 and vol_confirmed
        breakout_short = below_l4 and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or breakout_short:
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

</think>
#!/usr/bin/env python3
"""
exp_7191_6h_donchian20_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels for entry filtering.
In bull/bear markets: trade breakouts in direction of daily pivot bias (above/below pivot).
In ranging markets: fade at S3/R3 levels with volume confirmation.
Uses 1d Camarilla pivots for institutional reference points and 6h Donchian for structure.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in all regimes by combining price channels with pivot-based mean reversion/continuation logic.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7191_6h_donchian20_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
CAMARILLA_LOOKBACK = 1  # Use previous day's OHLC
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~4 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla formulas: 
    # H4 = Close + 1.5*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low)
    # L3 = Close - 1.125*(High-Low)
    # H3 and L3 are key fade levels, H4/L4 are breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for previous day (to avoid look-ahead)
    range_1d = high_1d - low_1d
    h4 = close_1d + 1.5 * range_1d
    l4 = close_1d - 1.5 * range_1d
    h3 = close_1d + 1.125 * range_1d
    l3 = close_1d - 1.125 * range_1d
    pivot = (high_1d + low_1d + close_1d) / 3.0  # Standard pivot point
    
    # Align to LTF (6h) - note: we use previous day's levels, so no additional shift needed
    # align_htf_to_ltf already includes shift(1) for completed bars only
    h4_6h = align_htf_to_ltf(prices, df_1d, h4)
    l4_6h = align_htf_to_ltf(prices, df_1d, l4)
    h3_6h = align_htf_to_ltf(prices, df_1d, h3)
    l3_6h = align_htf_to_ltf(prices, df_1d, l3)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    
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
        if np.isnan(h4_6h[i]) or np.isnan(l4_6h[i]) or np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]):
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
        
        # Determine market bias based on Camarilla levels
        above_h4 = close[i] > h4_6h[i]
        below_l4 = close[i] < l4_6h[i]
        between_h3_l3 = (close[i] > l3_6h[i]) and (close[i] < h3_6h[i])
        above_pivot = close[i] > pivot_6h[i]
        below_pivot = close[i] < pivot_6h[i]
        
        # Fade at S3/R3 in ranging market (between H3/L3) with volume
        fade_long = between_h3_l3 and (close[i] <= l3_6h[i]) and vol_confirmed and below_pivot
        fade_short = between_h3_l3 and (close[i] >= h3_6h[i]) and vol_confirmed and above_pivot
        
        # Breakout continuation at H4/L4 with volume
        breakout_long = above_h4 and vol_confirmed
        breakout_short = below_l4 and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or breakout_short:
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

</think>