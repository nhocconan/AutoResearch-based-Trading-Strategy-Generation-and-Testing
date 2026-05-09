#!/usr/bin/env python3
# Hypothesis: 12h Bollinger Band breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB, 1d EMA50 is rising, and volume > 1.5x 20-period average
# Short when price breaks below lower BB, 1d EMA50 is falling, and volume > 1.5x 20-period average
# Exit when price crosses back inside Bollinger Bands OR 1d EMA50 direction reverses
# Position size: 0.25 to balance return and drawdown
# Bollinger Bands capture volatility breakouts; EMA50 filters trend direction; volume confirms momentum
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns

name = "12h_Bollinger_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = (sma + bb_std * std).values
    lower_band = (sma - bb_std * std).values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, bb_period)  # Need enough data for EMA50 and BB
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper BB, 1d EMA50 rising, volume spike
            if (close[i] > upper_band[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB, 1d EMA50 falling, volume spike
            elif (close[i] < lower_band[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below lower BB OR 1d EMA50 turns falling
            if (close[i] < lower_band[i]) or (ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper BB OR 1d EMA50 turns rising
            if (close[i] > upper_band[i]) or (ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals