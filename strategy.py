#!/usr/bin/env python3
"""
exp_7407_6d_pivot3levels_volatility_breakout_v1
Hypothesis: 6-hour pivot breakout with 1-day/1-week trend filter and volume confirmation.
Uses daily/weekly pivot levels (R3/S3, R4/S4) for breakout entries, with trend filter
from 1d EMA and 1w EMA to avoid counter-trend trades. Volume confirms breakout strength.
Designed for low trade frequency (target: 75-200 total over 4 years) to minimize fee drag.
Works in bull/bear via multi-timeframe EMA filters: only long when above both EMAs,
short when below both EMAs.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7407_6d_pivot3levels_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 10  # days for weekly pivot
VOL_MA_PERIOD = 20
VOL_BREAKOUT_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # 3 days max hold

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d and 1w EMAs for trend filter
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to LTF (6h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily pivots from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot point: (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    r3 = high_1d + 2 * (pp - low_1d)
    s3 = low_1d - 2 * (high_1d - pp)
    r4 = pp + 3 * (high_1d - low_1d)
    s4 = pp - 3 * (high_1d - low_1d)
    
    # Align pivot levels to LTF
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(PIVOT_LOOKBACK, 50, 200, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]):
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
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BREAKOUT_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on 1d and 1w EMAs
        above_ema = close[i] > ema_1d_aligned[i] and close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i] and close[i] < ema_1w_aligned[i]
        
        # Breakout entries at R4/S4 with volume (continuation)
        breakout_long = above_ema and (close[i] > r4_aligned[i]) and vol_confirmed
        breakout_short = below_ema and (close[i] < s4_aligned[i]) and vol_confirmed
        
        # Mean reversion entries at R3/S3 with volume (fade)
        fade_long = not above_ema and not below_ema and (close[i] < r3_aligned[i]) and vol_confirmed
        fade_short = not above_ema and not below_ema and (close[i] > s3_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_long:
                signals[i] = SIGNAL_SIZE * 0.5  # smaller size for mean reversion
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short:
                signals[i] = -SIGNAL_SIZE * 0.5  # smaller size for mean reversion
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE if abs(position) == 1 else position * SIGNAL_SIZE * 0.5
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7407_6d_pivot3levels_volatility_breakout_v1
Hypothesis: 6-hour pivot breakout with 1-day/1-week trend filter and volume confirmation.
Uses daily/weekly pivot levels (R3/S3, R4/S4) for breakout entries, with trend filter
from 1d EMA and 1w EMA to avoid counter-trend trades. Volume confirms breakout strength.
Designed for low trade frequency (target: 75-200 total over 4 years) to minimize fee drag.
Works in bull/bear via multi-timeframe EMA filters: only long when above both EMAs,
short when below both EMAs.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7407_6d_pivot3levels_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 10  # days for weekly pivot
VOL_MA_PERIOD = 20
VOL_BREAKOUT_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # 3 days max hold

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d and 1w EMAs for trend filter
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to LTF (6h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily pivots from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot point: (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    r3 = high_1d + 2 * (pp - low_1d)
    s3 = low_1d - 2 * (high_1d - pp)
    r4 = pp + 3 * (high_1d - low_1d)
    s4 = pp - 3 * (high_1d - low_1d)
    
    # Align pivot levels to LTF
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(PIVOT_LOOKBACK, 50, 200, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]):
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
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BREAKOUT_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on 1d and 1w EMAs
        above_ema = close[i] > ema_1d_aligned[i] and close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i] and close[i] < ema_1w_aligned[i]
        
        # Breakout entries at R4/S4 with volume (continuation)
        breakout_long = above_ema and (close[i] > r4_aligned[i]) and vol_confirmed
        breakout_short = below_ema and (close[i] < s4_aligned[i]) and vol_confirmed
        
        # Mean reversion entries at R3/S3 with volume (fade)
        fade_long = not above_ema and not below_ema and (close[i] < r3_aligned[i]) and vol_confirmed
        fade_short = not above_ema and not below_ema and (close[i] > s3_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_long:
                signals[i] = SIGNAL_SIZE * 0.5  # smaller size for mean reversion
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short:
                signals[i] = -SIGNAL_SIZE * 0.5  # smaller size for mean reversion
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE if abs(position) == 1 else position * SIGNAL_SIZE * 0.5
    
    return signals