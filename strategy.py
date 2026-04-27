#!/usr/bin/env python3
# 12h_RSI_200_Trend_With_Volume_Confirmation
# Strategy: Long when RSI(14) < 30 and price > 200-period EMA with volume confirmation; short when RSI > 70 and price < 200 EMA with volume confirmation.
# Exit on opposite RSI extreme (RSI > 70 for long exit, RSI < 30 for short exit).
# Uses 1d timeframe for EMA200 trend filter to align with longer-term trend.
# Designed for 12-37 trades/year on 12h timeframe to minimize fee drag while capturing mean reversion in both bull and bear markets.

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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: RSI < 30 (oversold), price above EMA200, volume confirmation
        if (rsi[i] < 30 and 
            close[i] > ema200_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: RSI > 70 (overbought), price below EMA200, volume confirmation
        elif (rsi[i] > 70 and 
              close[i] < ema200_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI reaches opposite extreme
        elif position == 1 and rsi[i] > 70:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi[i] < 30:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_RSI_200_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0