# %pip install pandas numpy
#!/usr/bin/env python3
"""
Experiment #7659: 6-hour Camarilla pivot reversal with 12-hour volume confirmation.
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) act as strong support/resistance.
In ranging markets (price between R3/S3), fade extreme touches with volume confirmation.
In trending markets (price breaks R4/S4), continue in breakout direction.
Uses 12h timeframe for trend filter and volume confirmation to reduce noise.
Target: 50-150 trades over 4 years (12-37/year) with tight entry conditions.
Works in both bull/bear markets via adaptive regime detection.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7659_6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close
    c = (high + low + close) / 3
    r4 = c + (range_val * 1.1 / 2)
    r3 = c + (range_val * 1.1 / 4)
    s3 = c - (range_val * 1.1 / 4)
    s4 = c - (range_val * 1.1 / 2)
    return r4, r3, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < PIVOT_LOOKBACK + VOLUME_MA_PERIOD + ATR_PERIOD:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivots
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    r4_12h = np.full(len(close_12h), np.nan)
    r3_12h = np.full(len(close_12h), np.nan)
    s3_12h = np.full(len(close_12h), np.nan)
    s4_12h = np.full(len(close_12h), np.nan)
    
    for i in range(PIVOT_LOOKBACK-1, len(close_12h)):
        r4, r3, s3, s4 = calculate_camarilla(
            high_12h[i], low_12h[i], close_12h[i]
        )
        r4_12h[i] = r4
        r3_12h[i] = r3
        s3_12h[i] = s3
        s4_12h[i] = s4
    
    # Align HTF levels to LTF
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 12h volume average
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(
        window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD
    ).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(
        span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD
    ).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(volume_ma_12h_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation (12h)
        volume_confirmed = volume[i] > (volume_ma_12h_aligned[i] * VOLUME_THRESHOLD)
        
        # Price levels
        r4 = r4_12h_aligned[i]
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        
        # Determine market regime based on price position
        in_range = (s3 <= close <= r3)  # Between S3 and R3
        above_range = close > r4        # Above R4
        below_range = close < s4        # Below S4
        
        # Fade extreme touches in range
        if in_range and volume_confirmed:
            # Near S3 - potential bounce long
            if low[i] <= s3 * 1.002 and close[i] > s3:  # Touched S3 and bouncing
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Near R3 - potential reversal short
            elif high[i] >= r3 * 0.998 and close[i] < r3:  # Touched R3 and rejecting
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        # Breakout continuation
        elif above_range and volume_confirmed:
            # Strong breakout above R4 - continue long
            if close[i] > r4 and (i == start or close[i-1] <= r4_12h_aligned[i-1]):
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        elif below_range and volume_confirmed:
            # Strong breakdown below S4 - continue short
            if close[i] < s4 and (i == start or close[i-1] >= s4_12h_aligned[i-1]):
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        else:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals