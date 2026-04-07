#!/usr/bin/env python3
"""
1d_bollinger_squeeze_breakout
Hypothesis: Bollinger Band squeeze followed by breakout with volume confirmation captures volatility breakouts.
Works in bull markets (continuation) and bear markets (reversals) by combining low volatility breakout with volume.
Daily timeframe ensures low trade frequency, reducing fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_bollinger_squeeze_breakout"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA for trend filter (21-period)
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Daily Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    
    # Bollinger Band Width (for squeeze detection)
    bb_width = (upper - lower) / sma
    # Squeeze: BB width below 20-period mean of BB width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(bb_period, n):
        # Skip if data not available
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(squeeze[i]) or np.isnan(ema_21_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_21_aligned[i]
        downtrend = close[i] < ema_21_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle band (SMA) or opposite band touch
            if close[i] < sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band (SMA) or opposite band touch
            if close[i] > sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout conditions: Bollinger squeeze breakout with volume
            if squeeze[i-1]:  # Was in squeeze yesterday
                vol_confirmed = volume[i] > vol_ma[i]
                if vol_confirmed:
                    # Breakout above upper band
                    if close[i] > upper[i]:
                        position = 1
                        signals[i] = 0.25
                    # Breakout below lower band
                    elif close[i] < lower[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals