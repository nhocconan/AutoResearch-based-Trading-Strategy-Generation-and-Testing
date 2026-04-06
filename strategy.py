#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 6h Williams Alligator with 1d trend filter.
# Goes long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) 
# and price above 1d EMA200 (bullish regime).
# Goes short when Alligator jaws < teeth < lips and price below 1d EMA200 (bearish regime).
# Williams Alligator identifies trend phases (sleeping, awakening, feeding) which work in both bull and bear markets.
# The 1d EMA200 filter ensures we only trade in the direction of the higher timeframe trend.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13791_6h_alligator_1d_ema200"
timeframe = "6h"
leverage = 1.0

# Parameters
SMMA_PERIOD_JAWS = 13  # Alligator jaws (blue)
SMMA_PERIOD_TEETH = 8   # Alligator teeth (red)
SMMA_PERIOD_LIPS = 5    # Alligator lips (green)
TREND_EMA_PERIOD = 200  # 1d EMA for trend filter
SIGNAL_SIZE = 0.25      # Position size (25% of capital)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(data, np.nan, dtype=float)
    if len(data) >= period:
        smma[period-1] = sma[period-1]
        for i in range(period, len(data)):
            smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

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
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    
    # Align 1d EMA200 to 6h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h data for Alligator calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams Alligator components using SMMA on median price
    median_price = (high + low) / 2.0
    
    jaws = calculate_smma(median_price, SMMA_PERIOD_JAWS)   # Blue line (13-period)
    teeth = calculate_smma(median_price, SMMA_PERIOD_TEETH) # Red line (8-period)
    lips = calculate_smma(median_price, SMMA_PERIOD_LIPS)   # Green line (5-period)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SMMA_PERIOD_JAWS, SMMA_PERIOD_TEETH, SMMA_PERIOD_LIPS, TREND_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr[i])):
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
        
        # Alligator alignment signals
        # Bullish alignment: jaws > teeth > lips (alligator eating with mouth up)
        bullish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        # Bearish alignment: jaws < teeth < lips (alligator eating with mouth down)
        bearish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        
        # Trend filter from 1d EMA200
        above_200ema = close[i] > ema_200_1d_aligned[i]
        below_200ema = close[i] < ema_200_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: bullish alignment + price above 1d EMA200
            if bullish_alignment and above_200ema:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: bearish alignment + price below 1d EMA200
            elif bearish_alignment and below_200ema:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or price crosses below 1d EMA200
            if bearish_alignment or below_200ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: bullish alignment or price crosses above 1d EMA200
            if bullish_alignment or above_200ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals