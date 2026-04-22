#!/usr/bin/env python3
"""
Hypothesis: 12-hour RSI mean-reversion with 1-day trend filter and volume confirmation.
Long when RSI < 30 (oversold) and price above 1-day EMA200 (bullish bias).
Short when RSI > 70 (overbought) and price below 1-day EMA200 (bearish bias).
Exit when RSI returns to neutral (40-60 range) or opposite extreme is reached.
Uses 1-day EMA200 to filter higher timeframe trend, reducing counter-trend trades.
Designed for low trade frequency by requiring RSI extremes + trend alignment.
Works in both bull and bear markets by following daily trend while using 12h RSI for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after enough data for RSI
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) AND price above 1-day EMA200 AND volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema200_1d_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) AND price below 1-day EMA200 AND volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema200_1d_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI returns to neutral (40-60) or becomes overbought (>70)
                if (rsi[i] >= 40 and rsi[i] <= 60) or rsi[i] > 70:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI returns to neutral (40-60) or becomes oversold (<30)
                if (rsi[i] >= 40 and rsi[i] <= 60) or rsi[i] < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_RSI_MeanReversion_1dEMA200_Trend_Volume"
timeframe = "12h"
leverage = 1.0