#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price reversal at 12h pivot levels with volume confirmation.
# Uses 12h Camarilla pivot points (R3/S3 for reversal, R4/S4 for breakout).
# In sideways markets: fade at R3/S3 (mean reversion). In trending markets: breakout at R4/S4.
# Volume filter ensures only significant moves are traded.
# Target: 80-150 trades over 4 years (20-38/year) to balance opportunity and cost.

name = "exp_13159_6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20  # periods for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivots(high, low, close):
    """Calculate Camarilla pivot levels"""
    H = high[-1]
    L = low[-1]
    C = close[-1]
    range_ = H - L
    if range_ == 0:
        return C, C, C, C, C, C, C, C
    R4 = C + (range_ * 1.1 / 2)
    R3 = C + (range_ * 1.1 / 4)
    R2 = C + (range_ * 1.1 / 6)
    R1 = C + (range_ * 1.1 / 12)
    S1 = C - (range_ * 1.1 / 12)
    S2 = C - (range_ * 1.1 / 6)
    S3 = C - (range_ * 1.1 / 4)
    S4 = C - (range_ * 1.1 / 2)
    return R4, R3, R2, R1, S1, S2, S3, S4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Pre-calculate 12h pivot levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Arrays to store pivot levels for each 12h bar
    r4_12h = np.full(len(close_12h), np.nan)
    r3_12h = np.full(len(close_12h), np.nan)
    s3_12h = np.full(len(close_12h), np.nan)
    s4_12h = np.full(len(close_12h), np.nan)
    
    # Calculate pivots for each 12h bar (starting from PIVOT_LOOKBACK)
    for i in range(PIVOT_LOOKBACK, len(close_12h)):
        # Use last PIVOT_LOOKBACK periods to calculate pivot
        H = np.max(high_12h[i-PIVOT_LOOKBACK:i])
        L = np.min(low_12h[i-PIVOT_LOOKBACK:i])
        C = close_12h[i-1]  # previous close for pivot calculation
        range_ = H - L
        if range_ > 0:
            r4_12h[i] = C + (range_ * 1.1 / 2)
            r3_12h[i] = C + (range_ * 1.1 / 4)
            s3_12h[i] = C - (range_ * 1.1 / 4)
            s4_12h[i] = C - (range_ * 1.1 / 2)
        else:
            r4_12h[i] = C
            r3_12h[i] = C
            s3_12h[i] = C
            s4_12h[i] = C
    
    # Align pivot levels to 6h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot levels not available
        if np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price levels
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        r4 = r4_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        
        # Mean reversion signals at R3/S3 (fade)
        mean_rev_long = volume_ok and close[i] <= s3 and (i == 0 or close[i-1] > s3)
        mean_rev_short = volume_ok and close[i] >= r3 and (i == 0 or close[i-1] < r3)
        
        # Breakout signals at R4/S4 (continuation)
        breakout_long = volume_ok and close[i] >= r4 and (i == 0 or close[i-1] < r4)
        breakout_short = volume_ok and close[i] <= s4 and (i == 0 or close[i-1] > s4)
        
        # Generate signals
        if position == 0:
            if mean_rev_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif mean_rev_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action at 12h Camarilla pivot levels with volume confirmation.
# Uses dynamic pivot calculation based on recent 12h price action.
# In ranging markets: fade at R3/S3 levels (mean reversion).
# In trending markets: breakout at R4/S4 levels (continuation).
# Volume filter ensures only significant, institutional-backed moves are traded.
# Dual approach (mean reversion + breakout) adapts to changing market regimes.
# Target: 100-180 trades over 4 years (25-45/year) for statistical significance.

name = "exp_13159_6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20  # 12h periods for pivot calculation (10 days)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivots(high, low, close):
    """Calculate Camarilla pivot levels from high, low, close"""
    H = high[-1]
    L = low[-1]
    C = close[-1]
    range_ = H - L
    if range_ == 0:
        return C, C, C, C, C, C, C, C
    R4 = C + (range_ * 1.1 / 2)
    R3 = C + (range_ * 1.1 / 4)
    R2 = C + (range_ * 1.1 / 6)
    R1 = C + (range_ * 1.1 / 12)
    S1 = C - (range_ * 1.1 / 12)
    S2 = C - (range_ * 1.1 / 6)
    S3 = C - (range_ * 1.1 / 4)
    S4 = C - (range_ * 1.1 / 2)
    return R4, R3, R2, R1, S1, S2, S3, S4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EMA of true range)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop - CRITICAL for performance
    df_12h = get_htf_data(prices, '12h')
    
    # Extract 12h OHLC arrays
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pre-calculate pivot levels for each 12h bar
    # Arrays initialized with NaN
    r4_12h = np.full(len(close_12h), np.nan)
    r3_12h = np.full(len(close_12h), np.nan)
    s3_12h = np.full(len(close_12h), np.nan)
    s4_12h = np.full(len(close_12h), np.nan)
    
    # Calculate pivots for each 12h bar using rolling window
    # Start after we have enough data for lookback period
    for i in range(PIVOT_LOOKBACK, len(close_12h)):
        # Use the last PIVOT_LOOKBACK periods to calculate pivot point
        # Highest high and lowest low in the lookback window
        window_high = np.max(high_12h[i-PIVOT_LOOKBACK:i])
        window_low = np.min(low_12h[i-PIVOT_LOOKBACK:i])
        # Use the close of the previous period for pivot calculation (standard practice)
        prev_close = close_12h[i-1]
        
        range_ = window_high - window_low
        if range_ > 0:  # Avoid division by zero
            r4_12h[i] = prev_close + (range_ * 1.1 / 2)
            r3_12h[i] = prev_close + (range_ * 1.1 / 4)
            s3_12h[i] = prev_close - (range_ * 1.1 / 4)
            s4_12h[i] = prev_close - (range_ * 1.1 / 2)
        else:
            # If no range, all levels equal to close
            r4_12h[i] = prev_close
            r3_12h[i] = prev_close
            s3_12h[i] = prev_close
            s4_12h[i] = prev_close
    
    # Align 12h pivot levels to 6h timeframe
    # align_htf_to_ltf handles the shift(1) to avoid look-ahead
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss calculation
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Initialize arrays
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (ensures all indicators are valid)
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    # Main processing loop
    for i in range(start, n):
        # Skip if pivot levels are not yet available (first PIVOT_LOOKBACK bars)
        if np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]):
            # Hold current position or stay flat
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stop loss conditions
        if position == 1:  # Long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation - ensures significant participation
        volume_ok = False
        if not np.isnan(volume_ma[i]) and volume_ma[i] > 0:
            volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Current price levels from aligned 12h pivots
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        r4 = r4_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        
        # Mean reversion signals: fade at R3/S3 levels
        # Long when price touches S3 and shows rejection (close above S3)
        mean_rev_long = volume_ok and (close[i] <= s3) and (i == 0 or close[i-1] > s3)
        # Short when price touches R3 and shows rejection (close below R3)
        mean_rev_short = volume_ok and (close[i] >= r3) and (i == 0 or close[i-1] < r3)
        
        # Breakout signals: continuation at R4/S4 levels
        # Long when price breaks above R4 with conviction
        breakout_long = volume_ok and (close[i] >= r4) and (i == 0 or close[i-1] < r4)
        # Short when price breaks below S4 with conviction
        breakout_short = volume_ok and (close[i] <= s4) and (i == 0 or close[i-1] > s4)
        
        # Generate trading signals
        if position == 0:  # No position - look for entry
            if mean_rev_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif mean_rev_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0  # No signal - stay flat
        elif position == 1:  # Long position - maintain
            signals[i] = SIGNAL_SIZE
        elif position == -1:  # Short position - maintain
            signals[i] = -SIGNAL_SIZE
    
    return signals