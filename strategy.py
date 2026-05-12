#!/usr/bin/env python3
"""
4h_1d_RSI_Divergence_Scalper
Hypothesis: 4-hour RSI divergence signals with 1-day trend filter and volume confirmation. 
RSI divergence identifies potential reversals in both bull and bear markets. 
Trend filter ensures alignment with higher timeframe momentum, and volume confirmation 
avoids false signals. Targets 20-40 trades/year to minimize fee drag.
"""

name = "4h_1d_RSI_Divergence_Scalper"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: >1.8x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d data for trend filter and divergence confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Need sufficient history for divergence detection
    for i in range(50, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 5:
                # Look for price lower low and RSI higher low over last 5 periods
                if (low[i] < low[i-5] and 
                    rsi[i] > rsi[i-5] and 
                    volume_spike[i] and 
                    close[i] > ema_50_1d_aligned[i]):
                    bullish_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 5:
                # Look for price higher high and RSI lower high over last 5 periods
                if (high[i] > high[i-5] and 
                    rsi[i] < rsi[i-5] and 
                    volume_spike[i] and 
                    close[i] < ema_50_1d_aligned[i]):
                    bearish_div = True
            
            if bullish_div:
                signals[i] = 0.25
                position = 1
            elif bearish_div:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought or trend reversal
            if (rsi[i] > 70 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or trend reversal
            if (rsi[i] < 30 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals