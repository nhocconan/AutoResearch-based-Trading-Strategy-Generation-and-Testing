#!/usr/bin/env python3
"""
1d_Williams_Alligator_Bullish_Crossover_1wTrend
Hypothesis: Williams Alligator on daily timeframe with 1-week trend filter and volume confirmation. 
Go long when the Alligator lips (SMA5) cross above the teeth (SMA8) with price above the jaw (SMA13) and 1w uptrend.
Go short when lips cross below teeth with price below jaw and 1w downtrend.
Uses daily timeframe to reduce overtrading while capturing medium-term trends in both bull and bear markets.
Volume confirmation ensures institutional participation. Target: 15-25 trades/year per symbol.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Williams Alligator components on daily timeframe
    # Jaw (SMA13) - slow line
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    # Teeth (SMA8) - middle line
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    # Lips (SMA5) - fast line
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume spike: current volume > 1.5x 20-period average (less strict for daily)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for Alligator lines to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_13[i]) or np.isnan(sma_8[i]) or np.isnan(sma_5[i]) or 
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions
        # Bullish setup: lips above teeth, price above jaw
        bullish_setup = (sma_5[i] > sma_8[i]) and (close[i] > sma_13[i])
        # Bearish setup: lips below teeth, price below jaw
        bearish_setup = (sma_5[i] < sma_8[i]) and (close[i] < sma_13[i])
        
        # Trend filter from weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = bullish_setup and volume_spike[i] and uptrend
        short_entry = bearish_setup and volume_spike[i] and downtrend
        
        # Exit on opposite Alligator crossover
        long_exit = bearish_setup and volume_spike[i]
        short_exit = bullish_setup and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Williams_Alligator_Bullish_Crossover_1wTrend"
timeframe = "1d"
leverage = 1.0