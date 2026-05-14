#!/usr/bin/env python3
# Hypothesis: 6h ATR-based breakout with 1d trend filter and volume confirmation.
# The strategy uses ATR volatility breakouts (price moving beyond ATR-based thresholds) 
# combined with 1d EMA trend direction to capture momentum in both bull and bear markets.
# Volume confirmation (1.5x 24-period average) filters false breakouts.
# ATR adapts to market volatility, making it effective across different market regimes.
# Targets 50-150 total trades over 4 years with discrete position sizing to minimize fee drag.

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility-based breakout levels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate breakout levels: upper = close + 1.5*ATR, lower = close - 1.5*ATR
    # Using prior close to avoid look-ahead
    upper_break = np.roll(close, 1) + 1.5 * atr
    lower_break = np.roll(close, 1) - 1.5 * atr
    # Set first value to NaN since we don't have prior close
    upper_break[0] = np.nan
    lower_break[0] = np.nan
    
    # Volume filter: volume > 1.5x 24-period average (4 days of 6h bars)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24, 14)  # Wait for EMA, volume MA, and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_break[i]) or 
            np.isnan(lower_break[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # ATR breakout conditions
        breakout_up = high[i] > upper_break[i]  # Break above upper level
        breakout_down = low[i] < lower_break[i]  # Break below lower level
        
        # Entry conditions with volume spike confirmation
        long_entry = uptrend and breakout_up and volume_spike[i]
        short_entry = downtrend and breakout_down and volume_spike[i]
        
        # Exit conditions: trend reversal or opposite breakout
        long_exit = (not uptrend) or breakout_down
        short_exit = (not downtrend) or breakout_up
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ATRBreakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0