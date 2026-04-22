#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band breakout with 12-hour trend filter and volume confirmation.
Long when price breaks above upper BB(20,2) and 12h EMA(50) is rising and volume > 20-period average.
Short when price breaks below lower BB(20,2) and 12h EMA(50) is falling and volume > 20-period average.
Exit when price returns to middle band (SMA20).
Bollinger Bands capture volatility expansion; 12h EMA ensures trend alignment; volume filter confirms institutional interest.
Works in bull markets (riding trends) and bear markets (catching reversals from extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    middle_band = sma
    
    # Load 12-hour data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(sma[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB, 12h EMA rising, volume above average
            if close[i] > upper_band[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB, 12h EMA falling, volume above average
            elif close[i] < lower_band[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle band
                if close[i] >= middle_band[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle band
                if close[i] <= middle_band[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BB_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0