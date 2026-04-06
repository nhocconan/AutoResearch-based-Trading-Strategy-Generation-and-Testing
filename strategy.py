#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels with 1-day trend filter and volume confirmation.
# Uses 1-day EMA for trend direction, 6-hour Camarilla pivot levels for reversal entries at R3/S3 and breakout continuation at R4/S4.
# Volume confirmation (1.5x average) ensures momentum behind moves.
# Designed for ~100 total trades over 4 years (25/year) to avoid fee drain.
# Works in bull (R4/S4 breakouts) and bear (R3/S3 reversals) markets.

name = "exp_13711_6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 8
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    # Standard Camarilla levels
    r4 = c + range_val * 1.1 * 1.166  # ~1.2826
    r3 = c + range_val * 1.1 * 1.0833  # ~1.1916
    r2 = c + range_val * 1.1 * 1.0     # ~1.1000
    r1 = c + range_val * 1.1 * 0.8333  # ~0.9166
    s1 = c - range_val * 1.1 * 0.8333  # ~0.9166
    s2 = c - range_val * 1.1 * 1.0     # ~1.1000
    s3 = c - range_val * 1.1 * 1.0833  # ~1.1916
    s4 = c - range_val * 1.1 * 1.166   # ~1.2826
    return r1, r2, r3, r4, s1, s2, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Camarilla and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to hold Camarilla levels for each day
    r1_1d = np.full(len(close_1d), np.nan)
    r2_1d = np.full(len(close_1d), np.nan)
    r3_1d = np.full(len(close_1d), np.nan)
    r4_1d = np.full(len(close_1d), np.nan)
    s1_1d = np.full(len(close_1d), np.nan)
    s2_1d = np.full(len(close_1d), np.nan)
    s3_1d = np.full(len(close_1d), np.nan)
    s4_1d = np.full(len(close_1d), np.nan)
    
    # Calculate Camarilla for each day
    for i in range(len(close_1d)):
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        r2_1d[i] = r2
        r3_1d[i] = r3
        r4_1d[i] = r4
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Camarilla levels
        r3 = r3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Entry signals
        # Long: R3/S3 reversal in uptrend OR R4/S4 breakout in trend direction
        long_reversal = volume_ok and above_ema and close[i] <= s3 and close[i-1] > s3
        long_breakout = volume_ok and above_ema and close[i] >= r4 and close[i-1] < r4
        
        # Short: R3/S3 reversal in downtrend OR R4/S4 breakdown in trend direction
        short_reversal = volume_ok and below_ema and close[i] >= r3 and close[i-1] < r3
        short_breakout = volume_ok and below_ema and close[i] <= s4 and close[i-1] > s4
        
        long_signal = long_reversal or long_breakout
        short_signal = short_reversal or short_breakout
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite signal or stop (stop already checked)
            # Exit on R3 break below or S4 breakdown
            if close[i] < r3 and close[i-1] >= r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite signal or stop (stop already checked)
            # Exit on S3 break above or R4 break above
            if close[i] > s3 and close[i-1] <= s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels with 1-day trend filter and volume confirmation.
# Uses 1-day EMA for trend direction, 6-hour Camarilla pivot levels for reversal entries at R3/S3 and breakout continuation at R4/S4.
# Volume confirmation (1.5x average) ensures momentum behind moves.
# Designed for ~100 total trades over 4 years (25/year) to avoid fee drain.
# Works in bull (R4/S4 breakouts) and bear (R3/S3 reversals) markets.

name = "exp_13711_6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 8
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    # Standard Camarilla levels
    r4 = c + range_val * 1.1 * 1.166  # ~1.2826
    r3 = c + range_val * 1.1 * 1.0833  # ~1.1916
    r2 = c + range_val * 1.1 * 1.0     # ~1.1000
    r1 = c + range_val * 1.1 * 0.8333  # ~0.9166
    s1 = c - range_val * 1.1 * 0.8333  # ~0.9166
    s2 = c - range_val * 1.1 * 1.0     # ~1.1000
    s3 = c - range_val * 1.1 * 1.0833  # ~1.1916
    s4 = c - range_val * 1.1 * 1.166   # ~1.2826
    return r1, r2, r3, r4, s1, s2, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Camarilla and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to hold Camarilla levels for each day
    r1_1d = np.full(len(close_1d), np.nan)
    r2_1d = np.full(len(close_1d), np.nan)
    r3_1d = np.full(len(close_1d), np.nan)
    r4_1d = np.full(len(close_1d), np.nan)
    s1_1d = np.full(len(close_1d), np.nan)
    s2_1d = np.full(len(close_1d), np.nan)
    s3_1d = np.full(len(close_1d), np.nan)
    s4_1d = np.full(len(close_1d), np.nan)
    
    # Calculate Camarilla for each day
    for i in range(len(close_1d)):
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        r2_1d[i] = r2
        r3_1d[i] = r3
        r4_1d[i] = r4
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Camarilla levels
        r3 = r3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Entry signals
        # Long: R3/S3 reversal in uptrend OR R4/S4 breakout in trend direction
        long_reversal = volume_ok and above_ema and close[i] <= s3 and close[i-1] > s3
        long_breakout = volume_ok and above_ema and close[i] >= r4 and close[i-1] < r4
        
        # Short: R3/S3 reversal in downtrend OR R4/S4 breakdown in trend direction
        short_reversal = volume_ok and below_ema and close[i] >= r3 and close[i-1] < r3
        short_breakout = volume_ok and below_ema and close[i] <= s4 and close[i-1] > s4
        
        long_signal = long_reversal or long_breakout
        short_signal = short_reversal or short_breakout
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite signal or stop (stop already checked)
            # Exit on R3 break below or S4 breakdown
            if close[i] < r3 and close[i-1] >= r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite signal or stop (stop already checked)
            # Exit on S3 break above or R4 break above
            if close[i] > s3 and close[i-1] <= s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>