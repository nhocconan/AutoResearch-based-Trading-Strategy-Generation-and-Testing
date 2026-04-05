#!/usr/bin/env python3
"""
Experiment #7755: 6-hour Camarilla Pivot Fade with 1-week Trend Filter.
Hypothesis: In range-bound markets (common in BTC/ETH bear phases), price tends to revert from Camarilla R3/S3 levels.
In trending markets, price breaks through R4/S4 with continuation. Uses 1-week EMA50 for trend filter (bull/bear).
Fade at R3/S3 in range, breakout at R4/S4 in trend. Targets 50-150 trades over 4 years.
Works in bull markets (fade at R3 in uptrend, breakout at R4) and bear markets (fade at S3 in downtrend, breakdown at S4).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7755_6h_camarilla1w_fade_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1   # Use previous day for Camarilla calculation
TREND_EMA = 50       # 1-week EMA for trend filter
SIGNAL_SIZE = 0.25   # Position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    pivot = (high + low + close) / 3
    r4 = pivot + (range_val * 1.1 / 2)
    r3 = pivot + (range_val * 1.1 / 4)
    r2 = pivot + (range_val * 1.1 / 6)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1-week for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_EMA, adjust=False, min_periods=TREND_EMA).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily OHLC for Camarilla pivots (need 1d data)
    df_1d = get_htf_data(prices, '1d')
    
    # Pre-calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to store Camarilla levels
    n_1d = len(df_1d)
    r3_1d = np.full(n_1d, np.nan)
    s3_1d = np.full(n_1d, np.nan)
    r4_1d = np.full(n_1d, np.nan)
    s4_1d = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        if i >= PIVOT_LOOKBACK:
            # Use previous day's OHLC for today's levels
            idx = i - PIVOT_LOOKBACK
            _, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
                high_1d[idx], low_1d[idx], close_1d[idx]
            )
            r3_1d[i] = r3
            s3_1d[i] = s3
            r4_1d[i] = r4
            s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h ATR for risk management
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan  # First bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA, PIVOT_LOOKBACK + 1) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]):
            # Hold current position if any
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
        
        # Determine market regime from 1-week EMA
        bull_trend = close[i] > ema_1w_aligned[i]   # price above 1w EMA50
        bear_trend = close[i] < ema_1w_aligned[i]   # price below 1w EMA50
        
        # Get current Camarilla levels (from previous day's calculation)
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Skip if levels not available
        if np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Fade at R3/S3 in ranging markets, breakout at R4/S4 in trending markets
        # In bull trend: fade at R3 (sell), breakout at R4 (buy)
        # In bear trend: fade at S3 (buy), breakdown at S4 (sell)
        
        long_entry = False
        short_entry = False
        
        if bull_trend:
            # In uptrend: look for breakout above R4 or fade from R3
            if close[i] > r4 and close[i-1] <= r4:  # Breakout above R4
                long_entry = True
            elif close[i] < r3 and close[i-1] >= r3:  # Rejection at R3 (fade)
                short_entry = True
        elif bear_trend:
            # In downtrend: look for breakdown below S4 or fade from S3
            if close[i] < s4 and close[i-1] >= s4:  # Breakdown below S4
                short_entry = True
            elif close[i] > s3 and close[i-1] <= s3:  # Rejection at S3 (fade)
                long_entry = True
        else:
            # In transition or ranging: fade at R3/S3
            if close[i] < r3 and close[i-1] >= r3:  # Rejection at R3
                short_entry = True
            elif close[i] > s3 and close[i-1] <= s3:  # Rejection at S3
                long_entry = True
        
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
Experiment #7755: 6-hour Camarilla Pivot Fade with 1-week Trend Filter.
Hypothesis: In range-bound markets (common in BTC/ETH bear phases), price tends to revert from Camarilla R3/S3 levels.
In trending markets, price breaks through R4/S4 with continuation. Uses 1-week EMA50 for trend filter (bull/bear).
Fade at R3/S3 in range, breakout at R4/S4 in trend. Targets 50-150 trades over 4 years.
Works in bull markets (fade at R3 in uptrend, breakout at R4) and bear markets (fade at S3 in downtrend, breakdown at S4).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7755_6h_camarilla1w_fade_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1   # Use previous day for Camarilla calculation
TREND_EMA = 50       # 1-week EMA for trend filter
SIGNAL_SIZE = 0.25   # Position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    pivot = (high + low + close) / 3
    r4 = pivot + (range_val * 1.1 / 2)
    r3 = pivot + (range_val * 1.1 / 4)
    r2 = pivot + (range_val * 1.1 / 6)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1-week for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_EMA, adjust=False, min_periods=TREND_EMA).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily OHLC for Camarilla pivots (need 1d data)
    df_1d = get_htf_data(prices, '1d')
    
    # Pre-calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to store Camarilla levels
    n_1d = len(df_1d)
    r3_1d = np.full(n_1d, np.nan)
    s3_1d = np.full(n_1d, np.nan)
    r4_1d = np.full(n_1d, np.nan)
    s4_1d = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        if i >= PIVOT_LOOKBACK:
            # Use previous day's OHLC for today's levels
            idx = i - PIVOT_LOOKBACK
            _, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
                high_1d[idx], low_1d[idx], close_1d[idx]
            )
            r3_1d[i] = r3
            s3_1d[i] = s3
            r4_1d[i] = r4
            s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h ATR for risk management
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan  # First bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA, PIVOT_LOOKBACK + 1) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]):
            # Hold current position if any
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
        
        # Determine market regime from 1-week EMA
        bull_trend = close[i] > ema_1w_aligned[i]   # price above 1w EMA50
        bear_trend = close[i] < ema_1w_aligned[i]   # price below 1w EMA50
        
        # Get current Camarilla levels (from previous day's calculation)
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Skip if levels not available
        if np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Fade at R3/S3 in ranging markets, breakout at R4/S4 in trending markets
        # In bull trend: fade at R3 (sell), breakout at R4 (buy)
        # In bear trend: fade at S3 (buy), breakdown at S4 (sell)
        
        long_entry = False
        short_entry = False
        
        if bull_trend:
            # In uptrend: look for breakout above R4 or fade from R3
            if close[i] > r4 and close[i-1] <= r4:  # Breakout above R4
                long_entry = True
            elif close[i] < r3 and close[i-1] >= r3:  # Rejection at R3 (fade)
                short_entry = True
        elif bear_trend:
            # In downtrend: look for breakdown below S4 or fade from S3
            if close[i] < s4 and close[i-1] >= s4:  # Breakdown below S4
                short_entry = True
            elif close[i] > s3 and close[i-1] <= s3:  # Rejection at S3 (fade)
                long_entry = True
        else:
            # In transition or ranging: fade at R3/S3
            if close[i] < r3 and close[i-1] >= r3:  # Rejection at R3
                short_entry = True
            elif close[i] > s3 and close[i-1] <= s3:  # Rejection at S3
                long_entry = True
        
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
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>