#!/usr/bin/env python3
"""
4h_RSI_Trend_With_4h_Trend_Filter
Hypothesis: RSI combined with trend filter and volume confirmation provides high-probability entries in both bull and bear markets. 
Long when RSI > 50 + price > EMA20 + volume > 1.5x average + 4h close > 4h EMA50.
Short when RSI < 50 + price < EMA20 + volume > 1.5x average + 4h close < 4h EMA50.
Exit on opposite RSI condition. Position size: ±0.25. Uses 4h timeframe with trend filter.
Designed to capture trends while avoiding false signals via EMA50 filter.
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
    
    # Calculate EMA20 for price trend
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate EMA50 for trend filter
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (10-period MA)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 50, 14, 10)  # EMA20, EMA50, RSI, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma10[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        # RSI-based conditions
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Price relative to EMA20
        price_above_ema20 = close[i] > ema20[i]
        price_below_ema20 = close[i] < ema20[i]
        
        # Trend filter: 4h close relative to EMA50
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        if position == 0:
            # Long: RSI bullish + price above EMA20 + volume filter + uptrend
            if rsi_bullish and price_above_ema20 and volume_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: RSI bearish + price below EMA20 + volume filter + downtrend
            elif rsi_bearish and price_below_ema20 and volume_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI turns bearish (< 50)
            if rsi_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI turns bullish (> 50)
            if rsi_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Trend_With_4h_Trend_Filter"
timeframe = "4h"
leverage = 1.0