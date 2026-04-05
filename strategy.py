#!/usr/bin/env python3
"""
Experiment #7991: 6-hour Camarilla Pivot Reversal with Volume Confirmation.
Hypothesis: Price reacting at Camarilla R3/S3 levels (mean reversion) or breaking 
through R4/S4 levels (breakout) on 6h timeframe, filtered by 1d trend direction 
and volume confirmation, captures reversals in ranging markets and continuations 
in trending markets. Works in both bull and bear regimes by adapting to price 
action at key institutional levels.
"""

from mtf_data import get_hrf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7991_6h_camarilla_pivot_rev_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
PIVOT_LOOKBACK = 24  # 1 day of 6h bars

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=uptrend, -1=downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Calculate Camarilla levels for each bar using prior day's range
    # We need to calculate pivot based on previous day's HLC
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    # For each 6h bar, get the prior day's HLC
    for i in range(PIVOT_LOOKBACK, n):
        # Get index of same time previous day (4 intervals back for 6h -> 24h)
        prev_day_idx = i - 4
        if prev_day_idx >= 0:
            # Previous day's high, low, close
            phigh = high[prev_day_idx]
            plow = low[prev_day_idx]
            pclose = close[prev_day_idx]
            
            # Calculate pivot and Camarilla levels
            pivot = (phigh + plow + pclose) / 3
            range_val = phigh - plow
            
            camarilla_r4[i] = pclose + CAMARILLA_MULTIPLIER * range_val
            camarilla_r3[i] = pclose + (CAMARILLA_MULTIPLIER / 2) * range_val
            camarilla_s3[i] = pclose - (CAMARILLA_MULTIPLIER / 2) * range_val
            camarilla_s4[i] = pclose - CAMARILLA_MULTIPLIER * range_val
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ma[i])):
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
        
        # Determine market bias from 1d trend
        uptrend = trend_1d_aligned[i] == 1
        downtrend = trend_1d_aligned[i] == -1
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Price levels
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        
        # Mean reversion at R3/S3 (fade extreme levels)
        # Breakout continuation at R4/S4 (break institutional levels)
        long_setup = (close[i] <= s3 and not np.isnan(s3)) or (close[i] >= r4 and not np.isnan(r4))
        short_setup = (close[i] >= r3 and not np.isnan(r3)) or (close[i] <= s4 and not np.isnan(s4))
        
        # Entry conditions with trend filter
        # In uptrend: look for longs at S3/S4, shorts only at R4 (breakdown)
        # In downtrend: look for shorts at R3/R4, longs only at S4 (breakdown)
        if uptrend:
            long_entry = long_setup and volume_confirmed
            short_entry = False  # Only take shorts on breakdown of R4 in uptrend
        elif downtrend:
            short_entry = short_setup and volume_confirmed
            long_entry = False  # Only take longs on breakdown of S3 in downtrend
        else:
            long_entry = long_setup and volume_confirmed
            short_entry = short_setup and volume_confirmed
        
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