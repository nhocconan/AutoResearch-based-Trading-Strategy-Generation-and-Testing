#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Volume-Weighted Trend Following with 4h/1d Confirmation
# Hypothesis: In both bull and bear markets, price moves with volume continue in the direction of the higher timeframe trend.
# Use 4h trend (EMA crossover) and 1d regime (price vs SMA200) for direction, 1h for entry with volume confirmation.
# Volume filter ensures only institutional participation triggers entries, reducing false signals.
# Target: 15-37 trades/year (60-150 over 4 years).

name = "1h_volume_weighted_trend_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA crossover (9 and 21)
    ema9_4h = pd.Series(close_4h).ewm(span=9, adjust=False).mean().values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    # Trend: 1 if EMA9 > EMA21, -1 if EMA9 < EMA21
    trend_4h = np.where(ema9_4h > ema21_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d SMA200
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    # Regime: 1 if price > SMA200 (bull), -1 if price < SMA200 (bear)
    regime_1d = np.where(close_1d > sma200_1d, 1, -1)
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(regime_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Require session and volume filter
        if not (session_filter[i] and vol_filter[i]):
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Require 4h trend and 1d regime to agree
        if trend_4h_aligned[i] == regime_1d_aligned[i]:
            signal_direction = trend_4h_aligned[i]  # 1 for long, -1 for short
            
            if position == 0:
                # Enter in direction of aligned trend/regime
                if signal_direction == 1:
                    position = 1
                    signals[i] = 0.20
                elif signal_direction == -1:
                    position = -1
                    signals[i] = -0.20
            elif position * signal_direction < 0:
                # Reverse position if trend/regime flips
                position = signal_direction
                signals[i] = 0.20 if signal_direction == 1 else -0.20
            elif position * signal_direction > 0:
                # Maintain position
                signals[i] = 0.20 if position == 1 else -0.20
            else:
                # Flat
                signals[i] = 0.0
        else:
            # Trend and regime disagree - exit if in position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals