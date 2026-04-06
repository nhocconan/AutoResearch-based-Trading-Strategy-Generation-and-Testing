#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaws/Teeth/Lips) with 1d trend filter and volume confirmation.
# The Alligator uses three SMAs (13,8,5) to identify trend and ranging markets.
# When the three lines are intertwined (jaws closed) = ranging market (chop).
# When they diverge (jaws open) = trending market.
# We take long when Lips > Teeth > Jaws (bullish alignment) and short when Lips < Teeth < Jaws (bearish).
# Add 1d EMA(50) trend filter: only take signals in direction of higher timeframe trend.
# Volume confirmation ensures institutional participation.
# Target: 12-37 trades/year by using strict alignment conditions + trend + volume filters.

name = "exp_13632_12h_alligator_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
ALLIGATOR_JAWS = 13   # 13-period SMMA (smoothed)
ALLIGATOR_TEETH = 8   # 8-period SMMA
ALLIGATOR_LIPS = 5    # 5-period SMMA
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    return pd.Series(data).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_slope = np.diff(ema_1d, prepend=ema_1d[0])  # slope approximation
    ema_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slope)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: three SMMA lines
    jaws = smma(close, ALLIGATOR_JAWS)   # Blue line (13)
    teeth = smma(close, ALLIGATOR_TEETH) # Red line (8)
    lips = smma(close, ALLIGATOR_LIPS)   # Green line (5)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAWS, ALLIGATOR_TEETH, ALLIGATOR_LIPS, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_1d_slope_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA slope
        uptrend = ema_1d_slope_aligned[i] > 0
        downtrend = ema_1d_slope_aligned[i] < 0
        
        # Alligator signals: aligned jaws/teeth/lips
        # Bullish: Lips > Teeth > Jaws (green above red above blue)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaws[i]
        # Bearish: Lips < Teeth < Jaws (green below red below blue)
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaws[i]
        
        # Generate signals
        if position == 0:
            if bullish_alignment and volume_ok and in_session and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_alignment and volume_ok and in_session and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish alignment or stop loss
            if bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on bullish alignment or stop loss
            if bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals