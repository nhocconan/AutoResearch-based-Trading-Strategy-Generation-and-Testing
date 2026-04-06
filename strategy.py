#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Pivot Points (Standard) for direction and 6h Donchian(20) breakouts for entries.
# Uses 1d pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with volume confirmation.
# Designed for ~100-150 total trades over 4 years (25-38/year) to avoid excessive fees.
# Works in bull (breakouts at R4/S4 with volume) and bear (breakdowns at R4/S4 with volume) markets.
# Target: 100-200 total trades, 0.25 position size, max DD < -50%.

name = "exp_13747_6h_pivot_standard_20donchian_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, R2, R3, R4, S1, S2, S3, S4"""
    P = (high + low + close) / 3.0
    R1 = 2*P - low
    S1 = 2*P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    R3 = high + 2*(P - low)
    S3 = low - 2*(high - P)
    R4 = R3 + (high - low)
    S4 = S3 - (high - low)
    return P, R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for pivots ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize pivot arrays
    P = np.full_like(close_1d, np.nan)
    R1 = np.full_like(close_1d, np.nan)
    R2 = np.full_like(close_1d, np.nan)
    R3 = np.full_like(close_1d, np.nan)
    R4 = np.full_like(close_1d, np.nan)
    S1 = np.full_like(close_1d, np.nan)
    S2 = np.full_like(close_1d, np.nan)
    S3 = np.full_like(close_1d, np.nan)
    S4 = np.full_like(close_1d, np.nan)
    
    # Calculate pivots for each day (skip first day as we need prior day)
    for i in range(1, len(high_1d)):
        P[i], R1[i], R2[i], R3[i], R4[i], S1[i], S2[i], S3[i], S4[i] = \
            calculate_pivot_points(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Align pivots to 6h timeframe
    P_6h = align_htf_to_ltf(prices, df_1d, P)
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Load 6h data for entries
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # ATR for stop loss (using 6h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # 6h Donchian channels
    donchian_high = pd.Series(high_6h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_6h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(P_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_6h[i])):
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
        
        # Volume confirmation (using 6h volume)
        volume_ok = volume_6h[i] > (volume_ma_6h[i] * VOLUME_THRESHOLD)
        
        # Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_low[i-1]):
            high_prev = donchian_high[i-1]
            low_prev = donchian_low[i-1]
            
            # Long signal: break above R4 with volume (breakout continuation)
            long_signal = volume_ok and close_6h[i] > R4_6h[i] and close_6h[i-1] <= R4_6h[i] and close_6h[i] > high_prev
            
            # Alternative long: mean reversion from S3 with volume
            long_signal_alt = volume_ok and close_6h[i] < S3_6h[i] and close_6h[i-1] >= S3_6h[i] and close_6h[i] < low_prev
            
            # Short signal: break below S4 with volume (breakdown continuation)
            short_signal = volume_ok and close_6h[i] < S4_6h[i] and close_6h[i-1] >= S4_6h[i] and close_6h[i] < low_prev
            
            # Alternative short: mean reversion from R3 with volume
            short_signal_alt = volume_ok and close_6h[i] > R3_6h[i] and close_6h[i-1] <= R3_6h[i] and close_6h[i] > high_prev
        else:
            long_signal = False
            long_signal_alt = False
            short_signal = False
            short_signal_alt = False
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif long_signal_alt:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal_alt:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite Donchian break or mean reversion signal
            if i > 0 and not np.isnan(donchian_low[i-1]) and not np.isnan(donchian_low[i]):
                low_prev = donchian_low[i-1]
                if close_6h[i] < low_prev and close_6h[i-1] >= low_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite Donchian break or mean reversion signal
            if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_high[i]):
                high_prev = donchian_high[i-1]
                if close_6h[i] > high_prev and close_6h[i-1] <= high_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals