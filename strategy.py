# 6h Camarilla Pivot + 12h Trend + Volume Confirmation
# Hypothesis: Camarilla levels (R3/S3, R4/S4) act as key support/resistance on 6h.
# Trend from 12h filters direction: only long in uptrend, short in downtrend.
# Volume ensures momentum behind moves. Works in bull via R4 breakouts, bear via S4 breakdowns.
# Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12439_6h_camarilla12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price for the period
    typical_price = (high + low + close) / 3.0
    # Range
    range_ = high - low
    
    # Camarilla levels
    r4 = typical_price + (range_ * 1.1 / 2)
    r3 = typical_price + (range_ * 1.1 / 4)
    r2 = typical_price + (range_ * 1.1 / 6)
    r1 = typical_price + (range_ * 1.1 / 12)
    
    s1 = typical_price - (range_ * 1.1 / 12)
    s2 = typical_price - (range_ * 1.1 / 6)
    s3 = typical_price - (range_ * 1.1 / 4)
    s4 = typical_price - (range_ * 1.1 / 2)
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend
    ema_12h = calculate_ema(df_12h['close'].values, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla levels from previous bar
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high, low, close)
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Volume confirmation (use current bar volume vs MA of previous bar)
        vol_ma_prev = volume_ma[i-1] if i > 0 and not np.isnan(volume_ma[i-1]) else 0
        volume_ok = volume[i] > (vol_ma_prev * VOLUME_THRESHOLD) if vol_ma_prev > 0 else False
        
        # Trend filter (12h)
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Camarilla conditions (use previous bar levels)
        r3_prev = r3[i-1] if i > 0 and not np.isnan(r3[i-1]) else 0
        r4_prev = r4[i-1] if i > 0 and not np.isnan(r4[i-1]) else 0
        s3_prev = s3[i-1] if i > 0 and not np.isnan(s3[i-1]) else 0
        s4_prev = s4[i-1] if i > 0 and not np.isnan(s4[i-1]) else 0
        
        # Long conditions: break above R3/R4 in uptrend
        long_break_r3 = close[i] > r3_prev and close[i-1] <= r3_prev
        long_break_r4 = close[i] > r4_prev and close[i-1] <= r4_prev
        long_entry = volume_ok and uptrend_12h and (long_break_r3 or long_break_r4)
        
        # Short conditions: break below S3/S4 in downtrend
        short_break_s3 = close[i] < s3_prev and close[i-1] >= s3_prev
        short_break_s4 = close[i] < s4_prev and close[i-1] >= s4_prev
        short_entry = volume_ok and downtrend_12h and (short_break_s3 or short_break_s4)
        
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