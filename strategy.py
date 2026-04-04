#!/usr/bin/env python3
"""
exp_6847_6h_camarilla1d_pivot_v1
Hypothesis: 6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4.
Camarilla pivots identify intraday support/resistance with statistical significance.
In ranging markets: fade extreme levels (R3/S3) with mean reversion.
In trending markets: break through R4/S4 signals continuation with volume confirmation.
Uses 1d HTF for pivot calculation to avoid look-ahead and align with actual daily sessions.
Designed for 6h timeframe to capture 12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to regime via price action at pivot levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6847_6h_camarilla1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # use previous day's OHLC
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8  # ~2 days (6h bars)
PIVOT_PERIOD = 10  # for adaptive pivot relevance

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = pp + range_1d * 1.1 / 2
    r3 = pp + range_1d * 1.1 / 4
    r2 = pp + range_1d * 1.1 / 6
    r1 = pp + range_1d * 1.1 / 12
    s1 = pp - range_1d * 1.1 / 12
    s2 = pp - range_1d * 1.1 / 6
    s3 = pp - range_1d * 1.1 / 4
    s4 = pp - range_1d * 1.1 / 2
    
    # Align HTF pivot levels to LTF (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if i < len(vol_ma) and not np.isnan(vol_ma[i]) else False
        
        # Fade at R3/S3 (mean reversion)
        fade_long = close[i] <= s3_aligned[i] and close[i] > s4_aligned[i] and vol_confirmed
        fade_short = close[i] >= r3_aligned[i] and close[i] < r4_aligned[i] and vol_confirmed
        
        # Breakout continuation at R4/S4 (trend following)
        breakout_long = close[i] > r4_aligned[i] and vol_confirmed
        breakout_short = close[i] < s4_aligned[i] and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
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
exp_6847_6h_camarilla1d_pivot_v1
Hypothesis: 6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4.
Camarilla pivots identify intraday support/resistance with statistical significance.
In ranging markets: fade extreme levels (R3/S3) with mean reversion.
In trending markets: break through R4/S4 signals continuation with volume confirmation.
Uses 1d HTF for pivot calculation to avoid look-ahead and align with actual daily sessions.
Designed for 6h timeframe to capture 12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to regime via price action at pivot levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6847_6h_camarilla1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # use previous day's OHLC
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8  # ~2 days (6h bars)
PIVOT_PERIOD = 10  # for adaptive pivot relevance

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = pp + range_1d * 1.1 / 2
    r3 = pp + range_1d * 1.1 / 4
    r2 = pp + range_1d * 1.1 / 6
    r1 = pp + range_1d * 1.1 / 12
    s1 = pp - range_1d * 1.1 / 12
    s2 = pp - range_1d * 1.1 / 6
    s3 = pp - range_1d * 1.1 / 4
    s4 = pp - range_1d * 1.1 / 2
    
    # Align HTF pivot levels to LTF (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if i < len(vol_ma) and not np.isnan(vol_ma[i]) else False
        
        # Fade at R3/S3 (mean reversion)
        fade_long = close[i] <= s3_aligned[i] and close[i] > s4_aligned[i] and vol_confirmed
        fade_short = close[i] >= r3_aligned[i] and close[i] < r4_aligned[i] and vol_confirmed
        
        # Breakout continuation at R4/S4 (trend following)
        breakout_long = close[i] > r4_aligned[i] and vol_confirmed
        breakout_short = close[i] < s4_aligned[i] and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
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