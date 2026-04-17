#!/usr/bin/env python3
"""
4h_MomentumBreakout_Scalper
Hypothesis: Momentum breakouts from price channels combined with volume confirmation and RSI filter work in both bull and bear markets by capturing strong directional moves while avoiding chop. Uses 4h timeframe with 1-day trend filter for higher reliability. Target: 20-40 trades/year to avoid fee drag.
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
    
    # Donchian channel (20-period)
    def donchian_channel(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    upper_channel, lower_channel = donchian_channel(high, low, 20)
    
    # RSI (14-period)
    def rsi(close, window):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean().values
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 14, 20, 50)  # Donchian, RSI, volume MA, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(rsi_vals[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Breakout conditions
        breakout_up = close[i] > upper_channel[i-1]  # Use previous bar's upper channel
        breakout_down = close[i] < lower_channel[i-1]  # Use previous bar's lower channel
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_vals[i] < 70
        rsi_not_oversold = rsi_vals[i] > 30
        
        if position == 0:
            # Long: upward breakout + volume filter + RSI not overbought + 1d uptrend
            if breakout_up and volume_filter and rsi_not_overbought and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume filter + RSI not oversold + 1d downtrend
            elif breakout_down and volume_filter and rsi_not_oversold and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower channel or RSI overbought
            if close[i] < lower_channel[i-1] or rsi_vals[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper channel or RSI oversold
            if close[i] > upper_channel[i-1] or rsi_vals[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MomentumBreakout_Scalper"
timeframe = "4h"
leverage = 1.0