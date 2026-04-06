#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining Donchian(20) breakout with 1d VWAP and volume confirmation.
# Goes long when price breaks above 12h Donchian upper band with volume > 2x MA and price above 1d VWAP,
# short when breaks below 12h Donchian lower band with volume and price below 1d VWAP.
# Uses ATR stop loss and exits on opposite band touch. Designed for 50-150 total trades over 4 years.
# VWAP provides dynamic support/resistance, Donchian gives clear breakout levels, volume confirms strength.

name = "exp_13832_12h_donchian20_1d_vwap_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VWAP_PERIOD = 1  # For daily VWAP, we use typical price over the day
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0  # Higher threshold for fewer, higher quality trades
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_vwap(high, low, close):
    """Calculate VWAP for given period (typical price * volume) / volume"""
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * close) / close  # Using close as volume proxy for VWAP calculation
    # Actually, we need to calculate properly: cumulative(typical_price * volume) / cumulative(volume)
    # But since we don't have volume in VWAP calculation here, we'll use a simplified version
    # For daily VWAP on intraday data, we need the actual daily VWAP values
    # Let's return typical price as placeholder, will be replaced by actual VWAP from 1d data
    return typical_price

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP: (sum of typical_price * volume) / (sum of volume) for each day
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    
    # Align 1d VWAP to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # 12h data for Donchian channels, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels on 12h data
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(volume_ma[i]):
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
        
        # Price relative to 1d VWAP
        above_vwap = close[i] > vwap_1d_aligned[i]
        below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Donchian breakout signals
        long_signal = volume_ok and above_vwap and close[i] > upper[i]
        short_signal = volume_ok and below_vwap and close[i] < lower[i]
        
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
            # Exit long on close below Donchian lower band
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above Donchian upper band
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals