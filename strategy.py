# Hypothesis: 4h Camarilla pivot reversals with volume confirmation and volatility filter.
# Uses daily pivots for institutional levels, volume spike for institutional participation,
# and ATR-based volatility filter to avoid choppy markets. Designed for mean-reversion
# at S3/R3 and breakout at S4/R4, which works in both trending and ranging markets.
# Target: 100-150 total trades over 4 years (25-38/year) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12743_4h_camarilla_pivot_volume_volatility_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
VOLATILITY_LOOKBACK = 20
VOLATILITY_THRESHOLD = 0.5  # ATR ratio threshold
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_atr_ma(high, low, close, period, ma_period):
    """Calculate ATR and its moving average"""
    atr = calculate_atr(high, low, close, period)
    atr_ma = pd.Series(atr).rolling(window=ma_period, min_periods=ma_period).mean().values
    return atr, atr_ma

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    r4 = pivot + (range_val * 1.1)
    s4 = pivot - (range_val * 1.1)
    return r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r3_1d, s3_1d, r4_1d, s4_1d = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr, atr_ma = calculate_atr_ma(high, low, close, ATR_PERIOD, VOLATILITY_LOOKBACK)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, VOLATILITY_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily pivot levels not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
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
        volume_ok = (not np.isnan(volume_ma[i])) and (volume[i] > (volume_ma[i] * VOLUME_THRESHOLD))
        
        # Volatility filter: avoid extremely low volatility (chop) and extremely high volatility (chaos)
        vol_ratio = atr[i] / atr_ma[i] if not np.isnan(atr_ma[i]) and atr_ma[i] > 0 else 1.0
        volatility_ok = vol_ratio > VOLATILITY_THRESHOLD
        
        # Fade at R3/S3, breakout at R4/S4
        fade_long = volume_ok and volatility_ok and close[i] <= s3_1d_aligned[i]  # fade at S3 (support)
        fade_short = volume_ok and volatility_ok and close[i] >= r3_1d_aligned[i]  # fade at R3 (resistance)
        breakout_long = volume_ok and volatility_ok and close[i] >= r4_1d_aligned[i]  # breakout above R4
        breakout_short = volume_ok and volatility_ok and close[i] <= s4_1d_aligned[i]  # breakdown below S4
        
        # Entry conditions
        long_entry = fade_long or breakout_long
        short_entry = fade_short or breakout_short
        
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