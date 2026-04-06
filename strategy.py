#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Williams Alligator + Elder Ray with 12h trend filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trend presence and direction.
# Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA.
# Strategy: Go long when Alligator is bullish (Lips > Teeth > Jaw) AND Bull Power > 0.
# Go short when Alligator is bearish (Lips < Teeth < Jaw) AND Bear Power < 0.
# Use 12h EMA(50) as trend filter: only take long when 12h trend up, short when down.
# This combines trend-following (Alligator) with momentum (Elder Ray) and avoids
# counter-trend trades. Works in both bull (strong uptrend signals) and bear
# (strong downtrend signals) markets. Target: 20-40 trades/year.

name = "exp_13619_6s_alligator_elder_12h_trend_v1"
timeframe = "6s"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed Median (13)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed Median (8)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed Median (5)
ELDER_EMA_PERIOD = 13       # EMA for Elder Ray
TREND_EMA_PERIOD = 50       # 12h EMA for trend filter
SIGNAL_SIZE = 0.25          # Position size (25%)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smma(data, period):
    """Smoothed Moving Average (SMMA) - used in Alligator"""
    sma = np.mean(data[:period])
    smma = np.full_like(data, np.nan, dtype=np.float64)
    smma[period-1] = sma
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_ema_simple(close, period):
    """Simple EMA calculation for arrays"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Average True Range using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 6s indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (13), Teeth (8), Lips (5) - all SMMA of median price
    median_price = (high + low) / 2
    jaw = calculate_smma(median_price, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_smma(median_price, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_smma(median_price, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA
    ema_elder = calculate_ema(close, ELDER_EMA_PERIOD)
    bull_power = high - ema_elder
    bear_power = low - ema_elder
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD, 
                ELDER_EMA_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_12h_slope_aligned[i]) or np.isnan(atr[i])):
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
        
        # Alligator conditions
        # Bullish: Lips > Teeth > Jaw
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish: Lips < Teeth < Jaw
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Trend filter from 12h EMA slope
        uptrend = ema_12h_slope_aligned[i] > 0
        downtrend = ema_12h_slope_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            # Long: Alligator bullish AND Bull Power positive AND uptrend
            if alligator_bullish and bull_power_positive and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: Alligator bearish AND Bear Power negative AND downtrend
            elif alligator_bearish and bear_power_negative and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bear Power becomes negative
            if alligator_bearish or bear_power_negative:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bull Power becomes positive
            if alligator_bullish or bull_power_positive:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals