#!/usr/bin/env python3
"""
Hypothesis: 1h Timeframe with 4h Trend Direction and Volume Confirmation.
Long when 4h close > 4h EMA50 (uptrend), 1h RSI < 30 (oversold), and volume > 1.5x 20-period average.
Short when 4h close < 4h EMA50 (downtrend), 1h RSI > 70 (overbought), and volume > 1.5x 20-period average.
Exit when 4h trend reverses or RSI returns to neutral (40-60).
Designed for low trade frequency by requiring trend alignment and extreme RSI readings.
Works in both bull and bear markets by following the 4h trend and fading extremes.
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
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 4h close for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: 4h uptrend, RSI oversold, volume confirmation
            if (close[i] > ema50_4h_aligned[i] and  # 4h trend proxy using 1h close vs 4h EMA
                rsi[i] < 30 and vol_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, RSI overbought, volume confirmation
            elif (close[i] < ema50_4h_aligned[i] and 
                  rsi[i] > 70 and vol_confirmed):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: 4h trend reverses or RSI returns to neutral
            exit_signal = False
            
            if position == 1:
                # Exit long: 4h downtrend or RSI >= 40
                if close[i] < ema50_4h_aligned[i] or rsi[i] >= 40:
                    exit_signal = True
            else:  # position == -1
                # Exit short: 4h uptrend or RSI <= 60
                if close[i] > ema50_4h_aligned[i] or rsi[i] <= 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hTrend_RSIExtremes_Volume"
timeframe = "1h"
leverage = 1.0