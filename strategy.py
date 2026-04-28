#!/usr/bin/env python3
"""
4h_RSI_Divergence_Trend_Filter_v1
Hypothesis: Use RSI divergence on 1h timeframe with 4h EMA trend filter and volume confirmation to capture reversals in both bull and bear markets.
Targets 20-40 trades/year by requiring multiple confirmations, reducing false signals and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for RSI calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(df_1h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Align 1h RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi_values)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    h4_uptrend = close > ema_50_4h_aligned
    h4_downtrend = close < ema_50_4h_aligned
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Bullish RSI divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 5:
            # Look for recent swing low in price
            if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i-3] and low[i] < low[i-4]:
                # Check if this is a lower low compared to 5 periods ago
                if low[i] < low[i-5]:
                    # Check if RSI is making a higher low
                    if rsi_aligned[i] > rsi_aligned[i-5] and rsi_aligned[i] < 40:
                        bullish_div = True
        
        # Bearish RSI divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 5:
            # Look for recent swing high in price
            if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i-3] and high[i] > high[i-4]:
                # Check if this is a higher high compared to 5 periods ago
                if high[i] > high[i-5]:
                    # Check if RSI is making a lower high
                    if rsi_aligned[i] < rsi_aligned[i-5] and rsi_aligned[i] > 60:
                        bearish_div = True
        
        # Entry conditions
        # Long: bullish RSI divergence + 4h uptrend + volume surge
        long_entry = bullish_div and h4_uptrend[i] and volume_surge[i]
        
        # Short: bearish RSI divergence + 4h downtrend + volume surge
        short_entry = bearish_div and h4_downtrend[i] and volume_surge[i]
        
        # Exit on opposite signal
        long_exit = bearish_div and volume_surge[i]
        short_exit = bullish_div and volume_surge[i]
        
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

name = "4h_RSI_Divergence_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0