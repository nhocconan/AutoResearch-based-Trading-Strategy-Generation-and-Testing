#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13907_6d_camarilla1d_pivot_fade_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20  # Lookback for pivot calculation
VOLUME_MA = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for previous period
    
    Formulas:
    P = (H + L + C) / 3
    R1 = C + (H - L) * 1.1/12
    R2 = C + (H - L) * 1.1/6
    R3 = C + (H - L) * 1.1/4
    R4 = C + (H - L) * 1.1/2
    S1 = C - (H - L) * 1.1/12
    S2 = C - (H - L) * 1.1/6
    S3 = C - (H - L) * 1.1/4
    S4 = C - (H - L) * 1.1/2
    """
    P = (high + low + close) / 3.0
    range_hl = high - low
    
    R4 = close + range_hl * 1.1 / 2.0
    R3 = close + range_hl * 1.1 / 4.0
    R2 = close + range_hl * 1.1 / 6.0
    R1 = close + range_hl * 1.1 / 12.0
    
    S1 = close - range_hl * 1.1 / 12.0
    S2 = close - range_hl * 1.1 / 6.0
    S3 = close - range_hl * 1.1 / 4.0
    S4 = close - range_hl * 1.1 / 2.0
    
    return P, R1, R2, R3, R4, S1, S2, S3, S4

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
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla for each day
    P_vals = np.zeros(len(high_1d))
    R1_vals = np.zeros(len(high_1d))
    R2_vals = np.zeros(len(high_1d))
    R3_vals = np.zeros(len(high_1d))
    R4_vals = np.zeros(len(high_1d))
    S1_vals = np.zeros(len(high_1d))
    S2_vals = np.zeros(len(high_1d))
    S3_vals = np.zeros(len(high_1d))
    S4_vals = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        P, R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        P_vals[i] = P
        R1_vals[i] = R1
        R2_vals[i] = R2
        R3_vals[i] = R3
        R4_vals[i] = R4
        S1_vals[i] = S1
        S2_vals[i] = S2
        S3_vals[i] = S3
        S4_vals[i] = S4
    
    # Align to 6h timeframe (shifted by 1 day for previous day's levels)
    P_1d = align_htf_to_ltf(prices, df_1d, P_vals)
    R1_1d = align_htf_to_ltf(prices, df_1d, R1_vals)
    R2_1d = align_htf_to_ltf(prices, df_1d, R2_vals)
    R3_1d = align_htf_to_ltf(prices, df_1d, R3_vals)
    R4_1d = align_htf_to_ltf(prices, df_1d, R4_vals)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1_vals)
    S2_1d = align_htf_to_ltf(prices, df_1d, S2_vals)
    S3_1d = align_htf_to_ltf(prices, df_1d, S3_vals)
    S4_1d = align_htf_to_ltf(prices, df_1d, S4_vals)
    
    # 6h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_PERIOD, VOLUME_MA) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(P_1d[i]) or np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or np.isnan(volume_ma[i]):
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
        
        # Fade at R3/S3 levels (mean reversion)
        near_R3 = abs(close[i] - R3_1d[i]) < (0.1 * atr[i])  # Within 0.1 ATR of R3
        near_S3 = abs(close[i] - S3_1d[i]) < (0.1 * atr[i])  # Within 0.1 ATR of S3
        
        # Breakout continuation at R4/S4 (trend following)
        breakout_R4 = close[i] > R4_1d[i]
        breakout_S4 = close[i] < S4_1d[i]
        
        # Entry signals
        long_signal = volume_ok and (near_S3 or breakout_R4)
        short_signal = volume_ok and (near_R3 or breakout_S4)
        
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
            # Exit long on R3 touch or S4 breakout failure
            if near_R3 or (close[i] < R4_1d[i] and breakout_R4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on S3 touch or R4 breakout failure
            if near_S3 or (close[i] > S4_1d[i] and breakout_S4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals