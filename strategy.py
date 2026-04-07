#!/usr/bin/env python3
"""
4h_ema20_rsi_volume
Hypothesis: On 4h timeframe, EMA20 identifies short-term trend. RSI < 30 or > 70 indicates oversold/overbought conditions with reversal potential when aligned with EMA20 direction. Volume confirmation filters weak signals. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend). Target: 60-150 trades over 4 years (15-38/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema20_rsi_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA20
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(ema20[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend and momentum conditions
        price_above_ema = close[i] > ema20[i]
        price_below_ema = close[i] < ema20[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (momentum fade) or price crosses below EMA20
            if rsi[i] > 50 or close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (momentum fade) or price crosses above EMA20
            if rsi[i] < 50 or close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Buy oversold in uptrend
                if rsi_oversold and price_above_ema:
                    position = 1
                    signals[i] = 0.25
                # Sell overbought in downtrend
                elif rsi_overbought and price_below_ema:
                    position = -1
                    signals[i] = -0.25
    
    return signals