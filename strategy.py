#!/usr/bin/env python3
"""
1d_weekly_rsi_divergence_v1
Hypothesis: Weekly RSI divergence + daily price action for mean reversion.
- Only trade when weekly RSI shows divergence (bullish/bearish) at extremes
- Bullish divergence: weekly RSI makes higher low while price makes lower low
- Bearish divergence: weekly RSI makes lower high while price makes higher high
- Enter on daily close in direction of divergence with volume confirmation
- Exit on opposite divergence or weekly RSI normalization
- Target: 15-25 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_divergence_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder smoothing"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data (same as input)
    df_1d = prices.copy()
    
    # Weekly RSI for divergence detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation (20-period average)
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(close[i-1])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: bearish divergence or RSI normalization
            if (rsi_1w_aligned[i] < 30 and 
                rsi_1w_aligned[i-1] >= 30):  # RSI crossed below 30
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: bullish divergence or RSI normalization
            if (rsi_1w_aligned[i] > 70 and 
                rsi_1w_aligned[i-1] <= 70):  # RSI crossed above 70
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 2:
                # Look for recent swing lows
                if low[i] < low[i-1] and low[i-1] > low[i-2]:  # Recent low
                    # Check if this is a significant low vs prior
                    lookback = min(10, i-1)
                    if low[i] == np.min(low[i-lookback:i]):
                        # Check RSI at this point vs prior low
                        if (not np.isnan(rsi_1w_aligned[i]) and 
                            not np.isnan(rsi_1w_aligned[i-1]) and
                            rsi_1w_aligned[i] > rsi_1w_aligned[i-1] and
                            rsi_1w_aligned[i] < 40):  # Oversold but improving
                            bullish_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 2:
                # Look for recent swing highs
                if high[i] > high[i-1] and high[i-1] > high[i-2]:  # Recent high
                    # Check if this is a significant high vs prior
                    lookback = min(10, i-1)
                    if high[i] == np.max(high[i-lookback:i]):
                        # Check RSI at this point vs prior high
                        if (not np.isnan(rsi_1w_aligned[i]) and 
                            not np.isnan(rsi_1w_aligned[i-1]) and
                            rsi_1w_aligned[i] < rsi_1w_aligned[i-1] and
                            rsi_1w_aligned[i] > 60):  # Overbought but deteriorating
                            bearish_div = True
            
            # Long entry: bullish divergence + volume confirmation
            if bullish_div and volume[i] > vol_ma[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish divergence + volume confirmation
            elif bearish_div and volume[i] > vol_ma[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals