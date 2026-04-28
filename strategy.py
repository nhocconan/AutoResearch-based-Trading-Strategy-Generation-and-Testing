#!/usr/bin/env python3
"""
12h_RSI_Overbought_Oversold_Trend
Hypothesis: On 12-hour timeframe, enter long when RSI crosses above 30 (oversold recovery) with 1d uptrend and volume confirmation, short when RSI crosses below 70 (overbought rejection) with 1d downtrend and volume confirmation. Uses RSI for mean reversion in ranging markets and trend filter to avoid counter-trend trades. Designed for low trade frequency (~10-25/year) to minimize fee decay in both bull and bear markets.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Trend: bullish when price > EMA50, bearish when price < EMA50
    d1_uptrend = close > ema_50_aligned
    d1_downtrend = close < ema_50_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # RSI crossover conditions
        rsi_cross_up = rsi_values[i] > 30 and rsi_values[i-1] <= 30
        rsi_cross_down = rsi_values[i] < 70 and rsi_values[i-1] >= 70
        
        # Entry conditions with 1d EMA50 trend alignment and volume surge
        long_entry = rsi_cross_up and d1_uptrend[i] and volume_surge[i]
        short_entry = rsi_cross_down and d1_downtrend[i] and volume_surge[i]
        
        # Exit on opposite RSI level
        long_exit = rsi_values[i] < 70  # Exit long when RSI < 70 (overbought)
        short_exit = rsi_values[i] > 30  # Exit short when RSI > 30 (oversold)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_RSI_Overbought_Oversold_Trend"
timeframe = "12h"
leverage = 1.0