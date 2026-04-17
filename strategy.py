#!/usr/bin/env python3
"""
12h_Choppy_Market_Reversion
Strategy: Mean reversion in choppy markets using RSI and Bollinger Bands.
Long: RSI < 30 and price < lower Bollinger Band in choppy market (BW > 50th percentile)
Short: RSI > 70 and price > upper Bollinger Band in choppy market
Exit: RSI crosses 50 or volatility regime shifts
Position size: 0.25
Designed to capture reversals in ranging markets while avoiding trending periods.
Timeframe: 12h
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
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Band Width for regime detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Percentile rank of BB width (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Choppy market regime: BB width above 50th percentile (range-bound)
    choppy_market = bb_width_percentile > 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(choppy_market[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        
        if position == 0:
            # Long: oversold + price below lower BB + choppy market
            if oversold and price_below_lower and choppy_market[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought + price above upper BB + choppy market
            elif overbought and price_above_upper and choppy_market[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: RSI crosses above 50 or market becomes trending
            if rsi[i] > 50 or not choppy_market[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 or market becomes trending
            if rsi[i] < 50 or not choppy_market[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Choppy_Market_Reversion"
timeframe = "12h"
leverage = 1.0