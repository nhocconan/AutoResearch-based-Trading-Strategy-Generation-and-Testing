#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13970_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Hypothesis: 1d Donchian(20) breakout in direction of 1w EMA(20) with volume confirmation.
# Long when price breaks above 20-day high with price > 1w EMA and volume > 1.5x average.
# Short when price breaks below 20-day low with price < 1w EMA and volume > 1.5x average.
# Exit when price returns to 20-day midpoint or opposite breakout occurs.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull (breaks above with volume) and bear (breaks below with volume).

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
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA calculation ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA(20)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d data for Donchian channels, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-day Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_1w_aligned[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Donchian breakout signals
        breakout_high = close[i] > high_max[i-1]  # break above 20-day high
        breakdown_low = close[i] < low_min[i-1]   # break below 20-day low
        
        # EMA filter
        price_above_ema = close[i] > ema_1w_aligned[i]
        price_below_ema = close[i] < ema_1w_aligned[i]
        
        # Entry signals
        long_signal = breakout_high and price_above_ema and volume_ok
        short_signal = breakdown_low and price_below_ema and volume_ok
        
        # Exit signals (return to midpoint or opposite breakout)
        exit_long = close[i] < donchian_mid[i] or breakdown_low
        exit_short = close[i] > donchian_mid[i] or breakout_high
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on return to midpoint or opposite breakout
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on return to midpoint or opposite breakout
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals