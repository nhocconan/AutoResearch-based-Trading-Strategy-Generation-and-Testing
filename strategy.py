#!/usr/bin/env python3
"""
exp_6671_6h_camarilla_pivot_1d_v1
Hypothesis: 6h Camarilla pivot breakout/mean reversion with 1-day pivot direction filter.
Uses 1-day Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) to trade
intraday 6h swings. In ranging markets (price between R3-S3), fade extremes toward pivot.
In trending markets (break R4/S4), continue in breakout direction with volume confirmation.
1-day trend filter avoids counter-trend trades. Designed for 6h timeframe to capture
swings while minimizing fee drag (~20-40 trades/year expected).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6671_6h_camarilla_pivot_1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 4  # ~1 day (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Camarilla levels (based on previous day's OHLC)
    # Camarilla formulas:
    # Pivot = (high_prev + low_prev + close_prev) / 3
    # Range = high_prev - low_prev
    # R4 = close_prev + Range * 1.1/2
    # R3 = close_prev + Range * 1.1/4
    # S3 = close_prev - Range * 1.1/4
    # S4 = close_prev - Range * 1.1/2
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 for lookback)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    
    r4 = prev_close + rng * 1.1 / 2.0
    r3 = prev_close + rng * 1.1 / 4.0
    s3 = prev_close - rng * 1.1 / 4.0
    s4 = prev_close - rng * 1.1 / 2.0
    
    # Align HTF levels to LTF (6h) with shift(1) for completed days only
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
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
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine market regime based on 1d pivot
        # Above R4 = strong uptrend, Below S4 = strong downtrend
        # Between R3-S3 = ranging market
        strong_uptrend = close[i] > r4_aligned[i]
        strong_downtrend = close[i] < s4_aligned[i]
        ranging_market = (close[i] >= s3_aligned[i]) and (close[i] <= r3_aligned[i])
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Mean reversion signals (in ranging market)
        long_mean_revert = ranging_market and (close[i] <= s3_aligned[i])
        short_mean_revert = ranging_market and (close[i] >= r3_aligned[i])
        
        # Breakout signals (in trending market)
        long_breakout = strong_uptrend and vol_confirmed
        short_breakout = strong_downtrend and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_mean_revert or long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_mean_revert or short_breakout:
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