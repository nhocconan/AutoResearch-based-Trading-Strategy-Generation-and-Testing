#!/usr/bin/env python3
"""
6h_RSI_Divergence_Momentum_1dTrend_Filter
Hypothesis: Use 6h RSI divergence with price momentum for early reversal signals, filtered by 1d EMA trend. 
Bullish divergence (price LL, RSI HL) in 1d uptrend = long. Bearish divergence (price HH, RSI LH) in 1d downtrend = short.
Exit on RSI mean reversion or trend filter failure. Designed for 6h to capture swings with low frequency.
"""

name = "6h_RSI_Divergence_Momentum_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate RSI on 6h
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate price momentum (rate of change over 3 periods)
    roc_period = 3
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, rsi_period+1, roc_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Check for RSI divergence (need at least 5 bars back)
        if i >= 5:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-5] and rsi[i] > rsi[i-5] and 
                roc[i] > 0 and  # positive momentum confirmation
                trend_1d_up and
                rsi[i] < 40):  # not overbought
                if position == 0:
                    signals[i] = 0.25
                    position = 1
                elif position == -1:
                    signals[i] = 0.0  # close short
                    position = 0
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (high[i] > high[i-5] and rsi[i] < rsi[i-5] and 
                  roc[i] < 0 and  # negative momentum confirmation
                  trend_1d_down and
                  rsi[i] > 60):  # not oversold
                if position == 0:
                    signals[i] = -0.25
                    position = -1
                elif position == 1:
                    signals[i] = 0.0  # close long
                    position = 0
        
        # Exit conditions
        if position == 1:
            # Exit long: RSI mean reversion or trend failure
            if (rsi[i] > 60 or  # overbought
                not trend_1d_up or  # trend filter failed
                (rsi[i] < rsi[i-1] and rsi[i-1] > 60)):  # RSI turning down from overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI mean reversion or trend failure
            if (rsi[i] < 40 or  # oversold
                not trend_1d_down or  # trend filter failed
                (rsi[i] > rsi[i-1] and rsi[i-1] < 40)):  # RSI turning up from oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals