#!/usr/bin/env python3
"""
1d_RSI_Divergence_MeanReversion
Hypothesis: Use daily RSI divergences (price making new high/low while RSI fails to confirm) to capture mean reversion in overbought/oversold conditions, filtered by weekly trend and volume confirmation. Works in both bull (sell overextended rallies) and bear (buy oversold bounces) markets. Targets low trade frequency (<20/year) to minimize fee drag.
"""

name = "1d_RSI_Divergence_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to daily timeframe (no extra delay needed)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Get weekly trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (20-day) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(rsi_aligned[i-1]) or 
            np.isnan(rsi_aligned[i-2]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-day average
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bear_div = (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                   rsi_aligned[i] < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2])
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bull_div = (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                   rsi_aligned[i] > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2])
        
        if position == 0:
            # LONG: bullish divergence + volume confirmation + price above weekly EMA50 (uptrend filter)
            if bull_div and vol_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish divergence + volume confirmation + price below weekly EMA50 (downtrend filter)
            elif bear_div and vol_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish divergence or price breaks below EMA50
            if bear_div or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish divergence or price breaks above EMA50
            if bull_div or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals