#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14032_12h_donchian20_1d_vol_t1"
timeframe = "12h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian upper and lower bands"""
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

def calculate_ema(values, span):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Donchian (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian channels (20-period)
    donchian_upper_1d, donchian_lower_1d = calculate_donchian(high_1d, low_1d, 20)
    
    # Daily ATR for volatility filter (14-period)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align daily indicators to 12h timeframe
    donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h data for price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper_1d_aligned[i]) or np.isnan(donchian_lower_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Volatility filter: only trade when ATR is above average
        vol_filter = atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[max(0, i-50):i]) if i >= 50 else False
        
        # Donchian breakout signals (using previous bar's bands)
        breakout_up = close[i] > donchian_upper_1d_aligned[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower_1d_aligned[i-1]  # break below previous lower band
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_ok and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1d_aligned[i])
            elif breakout_down and volume_ok and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal at opposite band
            if close[i] <= stop_price or close[i] < donchian_lower_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reversal at opposite band
            if close[i] >= stop_price or close[i] > donchian_upper_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals