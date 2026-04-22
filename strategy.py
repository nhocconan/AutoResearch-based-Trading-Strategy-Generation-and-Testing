#!/usr/bin/env python3

"""
Hypothesis: 1-hour RSI mean reversion with 1-day trend filter and volume confirmation.
Only take long positions when RSI < 30 (oversold) and 1-day EMA50 is rising (bullish trend),
or short positions when RSI > 70 (overbought) and 1-day EMA50 is falling (bearish trend).
Requires volume > 1.5x 20-period average for confirmation. Uses 1-day trend to avoid
counter-trend trades in strong trends, improving win rate in both bull and bear markets.
Designed for low trade frequency (15-35 trades/year) by requiring multiple confirmations:
RSI extreme, trend alignment, and volume spike.
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
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend direction from 1-day EMA50 slope
        ema50_rising = ema50_1d_aligned[i] > ema50_1d_aligned[i-1]
        ema50_falling = ema50_1d_aligned[i] < ema50_1d_aligned[i-1]
        
        if position == 0:
            # Long: RSI oversold + 1-day EMA50 rising + volume spike
            if rsi[i] < 30 and ema50_rising and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + 1-day EMA50 falling + volume spike
            elif rsi[i] > 70 and ema50_falling and vol_spike:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 40 or 1-day EMA50 starts falling
                if rsi[i] >= 40 or not ema50_rising:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 60 or 1-day EMA50 starts rising
                if rsi[i] <= 60 or not ema50_falling:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_MeanReversion_1dEMA50Trend_Volume"
timeframe = "1h"
leverage = 1.0