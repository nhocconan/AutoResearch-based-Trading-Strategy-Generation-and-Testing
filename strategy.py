#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 on daily close for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: Bull Power > 0 (bullish momentum) + RSI > 50 (momentum) + volume
        long_signal = volume_confirmed and (bull_power_6h[i] > 0) and (rsi[i] > 50)
        
        # Short conditions: Bear Power < 0 (bearish momentum) + RSI < 50 (momentum) + volume
        short_signal = volume_confirmed and (bear_power_6h[i] < 0) and (rsi[i] < 50)
        
        # Exit when momentum diverges: Bull Power turns negative for long, Bear Power turns positive for short
        exit_long = position == 1 and bull_power_6h[i] <= 0
        exit_short = position == -1 and bear_power_6h[i] >= 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s Elder Ray momentum strategy with RSI confirmation and volume filter.
# Uses daily Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure
# market momentum relative to the 13-period EMA. Enters long when Bull Power > 0
# indicating bullish momentum, RSI > 50 confirming upward momentum, and volume > 1.5x average.
# Enters short when Bear Power < 0 indicating bearish momentum, RSI < 50, and volume confirmation.
# Exits when momentum diverges (Bull Power <= 0 for longs, Bear Power >= 0 for shorts).
# This approach works in both bull and bear markets by capturing momentum shifts.
# Position size: 0.25 to manage risk. Target: 50-150 trades over 4 years (12-37/year).