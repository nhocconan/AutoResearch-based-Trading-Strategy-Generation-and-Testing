#!/usr/bin/env python3
"""
Strategy: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above upper band, EMA50 rising, and volume > 1.5x 20-bar average.
Short when price breaks below lower band, EMA50 falling, and volume > 1.5x 20-bar average.
Exit when price crosses opposite band or EMA50 trend reverses.
Designed for low trade frequency and strong trend following with volume confirmation.
Works in bull markets via breakouts and bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_20
    lower_band = low_20
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band, EMA50 rising, volume confirmation
            if (close[i] > upper_band[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, EMA50 falling, volume confirmation
            elif (close[i] < lower_band[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below lower band OR EMA50 trend turns down
                if (close[i] < lower_band[i] or 
                    ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above upper band OR EMA50 trend turns up
                if (close[i] > upper_band[i] or 
                    ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0