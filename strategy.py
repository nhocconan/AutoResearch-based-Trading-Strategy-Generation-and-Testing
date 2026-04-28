#!/usr/bin/env python3
# Hypothesis: 4h Bollinger Bands squeeze breakout with 1d trend filter and volume confirmation.
# Uses Bollinger Bands width to identify low volatility periods (squeeze), then breaks out in the direction
# of the 1d EMA(50) trend. Volume confirmation (1.5x 20-period average) filters false breakouts.
# Designed for 4h timeframe with ~50-150 total trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std_dev * bb_std)
    lower_band = sma - (bb_std_dev * bb_std)
    bb_width = upper_band - lower_band
    
    # Bollinger Band squeeze: width below 50-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20)  # Wait for BB, EMA, and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(bb_std_dev[i]) or np.isnan(bb_width[i]) or
            np.isnan(bb_width_ma[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: breakout from squeeze in trend direction with volume
        long_breakout = close[i] > upper_band[i]
        short_breakout = close[i] < lower_band[i]
        
        long_entry = squeeze[i] and long_breakout and uptrend and volume_confirm[i]
        short_entry = squeeze[i] and short_breakout and downtrend and volume_confirm[i]
        
        # Exit conditions: opposite breakout or loss of trend
        long_exit = (close[i] < sma[i]) or (not uptrend)
        short_exit = (close[i] > sma[i]) or (not downtrend)
        
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

name = "4h_BollingerSqueeze_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0