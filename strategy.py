#!/usr/bin/env python3
"""
6h_triple_ema_reversion_v1
Hypothesis: On 6-hour timeframe, price reverts to the mean during overextended moves.
Uses triple EMA (8,21,50) alignment with RSI extreme and Bollinger Band width filter.
Long when EMA8 > EMA21 > EMA50 (bullish alignment) AND RSI < 30 AND BB width < 0.05 (low volatility).
Short when EMA8 < EMA21 < EMA50 (bearish alignment) AND RSI > 70 AND BB width < 0.05.
Exit when EMA8 crosses back toward EMA21.
Designed for low-frequency, high-conviction trades in ranging markets with volatility filter to avoid chop.
Works in both bull/bear markets as it fades extremes during low volatility regardless of trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_triple_ema_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMAs
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Band width (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(50, 20), n):
        # Skip if data not available
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(rsi[i]) or np.isnan(bb_width[i])):
            signals[i] = 0.0
            continue
            
        # Low volatility filter (avoid choppy markets)
        vol_filter = bb_width[i] < 0.05
        
        if position == 1:  # Long position
            # Exit: EMA8 crosses back below EMA21 (mean reversion)
            if ema8[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA8 crosses back above EMA21 (mean reversion)
            if ema8[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with low volatility filter
            if vol_filter:
                # Bullish EMA alignment + RSI oversold
                if (ema8[i] > ema21[i] > ema50[i] and rsi[i] < 30):
                    position = 1
                    signals[i] = 0.25
                # Bearish EMA alignment + RSI overbought
                elif (ema8[i] < ema21[i] < ema50[i] and rsi[i] > 70):
                    position = -1
                    signals[i] = -0.25
    
    return signals