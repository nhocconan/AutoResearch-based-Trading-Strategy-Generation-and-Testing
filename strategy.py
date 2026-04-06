#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot strategy using 1d pivot levels.
# Go long when price retraces to S1/S2 support in uptrend (price > 1w EMA200),
# go short when price retraces to R1/R2 resistance in downtrend (price < 1w EMA200).
# Uses volume confirmation to avoid false signals.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Camarilla levels provide precise reversal points in ranging markets,
# while weekly EMA filter ensures we trade with the higher timeframe trend.

name = "exp_13855_6h_camarilla1d_ewma200_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
EMA_WEEKLY_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    # Resistance levels
    R1 = pivot + (range_val * 1.1 / 12)
    R2 = pivot + (range_val * 1.1 / 6)
    R3 = pivot + (range_val * 1.1 / 4)
    R4 = pivot + (range_val * 1.1 / 2)
    # Support levels
    S1 = pivot - (range_val * 1.1 / 12)
    S2 = pivot - (range_val * 1.1 / 6)
    S3 = pivot - (range_val * 1.1 / 4)
    S4 = pivot - (range_val * 1.1 / 2)
    return R1, R2, R3, R4, S1, S2, S3, S4

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
    
    # Load 1d data for Camarilla calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Load 1w data for EMA trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_WEEKLY_PERIOD)
    ema_1w_6h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h data for price, volume, and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_WEEKLY_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(volume_ma[i])):
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
        
        # Trend direction from 1w EMA
        above_weekly_ema = close[i] > ema_1w_6h[i]
        below_weekly_ema = close[i] < ema_1w_6h[i]
        
        # Camarilla retracement signals with volume confirmation
        # Long: price near S1/S2 support in uptrend
        long_signal = volume_ok and above_weekly_ema and (
            close[i] <= S1_6h[i] * 1.005 or  # Allow small buffer
            close[i] <= S2_6h[i] * 1.005
        )
        
        # Short: price near R1/R2 resistance in downtrend
        short_signal = volume_ok and below_weekly_ema and (
            close[i] >= R1_6h[i] * 0.995 or  # Allow small buffer
            close[i] >= R2_6h[i] * 0.995
        )
        
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
            # Exit long on close above R1 (take profit) or below S3 (stop)
            if close[i] >= R1_6h[i] or close[i] <= S3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close below S1 (take profit) or above R3 (stop)
            if close[i] <= S1_6h[i] or close[i] >= R3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals