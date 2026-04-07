#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_trend_volume_v1
Hypothesis: RSI pullback on 4h with 1-day trend filter and volume confirmation.
In bullish regime (price > 1d EMA200), enter long on RSI(14) pullback from oversold (<30) with volume spike.
In bearish regime (price < 1d EMA200), enter short on RSI(14) pullback from overbought (>70) with volume spike.
Uses RSI mean reversion within the trend, filtered by 1-day trend and volume confirmation.
Designed for 20-40 trades/year on 4h timeframe with clear entry/exit rules that work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral or trend turns bearish
            if rsi[i] >= 50 or bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral or trend turns bullish
            if rsi[i] <= 50 or bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI oversold with volume confirmation and bullish trend
            if rsi_oversold and vol_confirmed and bullish_trend:
                position = 1
                signals[i] = 0.25
            # Short: RSI overbought with volume confirmation and bearish trend
            elif rsi_overbought and vol_confirmed and bearish_trend:
                position = -1
                signals[i] = -0.25
    
    return signals