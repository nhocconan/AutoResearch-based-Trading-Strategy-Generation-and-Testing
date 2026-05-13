#!/usr/bin/env python3
"""
4h_Choppiness_Index_MeanReversion
Hypothesis: Mean-reversion on 4h timeframe using Choppiness Index to identify range-bound markets (chop > 61.8).
Enter long at lower Bollinger Band with RSI < 30, short at upper Bollinger Band with RSI > 70.
Exit when price crosses Bollinger middle band or RSI reverts to neutral.
Designed for low trade frequency in both bull and bear markets by combining regime filter with mean reversion.
"""

name = "4h_Choppiness_Index_MeanReversion"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Choppiness Index (14-period)
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).mean()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.fillna(50).values  # neutral when undefined
    
    # Bollinger Bands (20, 2.0)
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma + 2.0 * std
    lower_bb = sma - 2.0 * std
    middle_bb = sma
    upper_bb_values = upper_bb.fillna(0).values
    lower_bb_values = lower_bb.fillna(0).values
    middle_bb_values = middle_bb.fillna(0).values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Only trade in ranging markets (chop > 61.8)
        if chop_values[i] <= 61.8:
            # Exit any position when market starts trending
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Price at lower BB with oversold RSI and volume confirmation
            if close[i] <= lower_bb_values[i] and rsi[i] < 30 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at upper BB with overbought RSI and volume confirmation
            elif close[i] >= upper_bb_values[i] and rsi[i] > 70 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above middle BB or RSI > 50
            if close[i] >= middle_bb_values[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below middle BB or RSI < 50
            if close[i] <= middle_bb_values[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals