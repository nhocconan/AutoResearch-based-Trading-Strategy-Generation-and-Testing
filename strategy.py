#!/usr/bin/env python3
"""
Experiment #10015: 6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Camarilla pivot levels provide strong support/resistance. Price rejection at R3/S3 with volume confirmation offers high-probability reversals, while breaks of R4/S4 indicate continuation. Works in ranging markets (reversions) and trending markets (breakouts). Volume filters reduce false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10015_6h_camarilla_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels from previous day's OHLC"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r3 = pivot + (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1)
    
    return r3, r4, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0  # First value has no previous close
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    r3_daily, r4_daily, s3_daily, s4_daily = calculate_camarilla_pivots(daily_high, daily_low, daily_close)
    
    # Align daily levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 20  # For volume MA
    
    for i in range(start, n):
        # Skip if daily levels not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Reversal at S3/R3 with volume
        near_s3 = low[i] <= s3_aligned[i] * 1.002 and close[i] > s3_aligned[i]  # Touched S3 and closed above
        near_r3 = high[i] >= r3_aligned[i] * 0.998 and close[i] < r3_aligned[i]  # Touched R3 and closed below
        
        # Breakout of S4/R4 for continuation
        break_s4 = close[i] < s4_aligned[i] and low[i] < s4_aligned[i]  # Closed below S4
        break_r4 = close[i] > r4_aligned[i] and high[i] > r4_aligned[i]  # Closed above R4
        
        # Entry conditions
        long_entry = near_s3 and volume_spike  # Buy the dip at S3 with volume
        short_entry = near_r3 and volume_spike  # Sell the rally at R3 with volume
        long_break = break_s4 and volume_spike  # Breakdown continuation
        short_break = break_r4 and volume_spike  # Breakout continuation
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_break:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_break:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>

#!/usr/bin/env python3
"""
Experiment #10015: 6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Camarilla pivot levels provide strong support/resistance. Price rejection at R3/S3 with volume confirmation offers high-probability reversals, while breaks of R4/S4 indicate continuation. Works in ranging markets (reversions) and trending markets (breakouts). Volume filters reduce false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10015_6h_camarilla_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels from previous day's OHLC"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r3 = pivot + (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1)
    
    return r3, r4, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0  # First value has no previous close
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    r3_daily, r4_daily, s3_daily, s4_daily = calculate_camarilla_pivots(daily_high, daily_low, daily_close)
    
    # Align daily levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 20  # For volume MA
    
    for i in range(start, n):
        # Skip if daily levels not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Reversal at S3/R3 with volume
        near_s3 = low[i] <= s3_aligned[i] * 1.002 and close[i] > s3_aligned[i]  # Touched S3 and closed above
        near_r3 = high[i] >= r3_aligned[i] * 0.998 and close[i] < r3_aligned[i]  # Touched R3 and closed below
        
        # Breakout of S4/R4 for continuation
        break_s4 = close[i] < s4_aligned[i] and low[i] < s4_aligned[i]  # Closed below S4
        break_r4 = close[i] > r4_aligned[i] and high[i] > r4_aligned[i]  # Closed above R4
        
        # Entry conditions
        long_entry = near_s3 and volume_spike  # Buy the dip at S3 with volume
        short_entry = near_r3 and volume_spike  # Sell the rally at R3 with volume
        long_break = break_s4 and volume_spike  # Breakdown continuation
        short_break = break_r4 and volume_spike  # Breakout continuation
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_break:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_break:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>