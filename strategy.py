#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d and 1w timeframes for regime and trend, 4h for entry.
# Uses 1d RSI(14) regime filter to identify overbought/oversold conditions, 1w EMA200 for long-term trend,
# and 4h Donchian(20) breakout with volume confirmation for entry.
# Designed to work in both bull and bear markets by combining trend-following with mean-reversion in extremes.
# Target: 20-50 trades per year to minimize fee drag and improve generalization.
name = "4h_RSIRegime_1wEMA200_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high of last 20 periods
    upper_4h = np.full(len(high_4h), np.nan)
    for i in range(20, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-20:i])
    
    # Lower band: lowest low of last 20 periods
    lower_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(low_4h)):
        lower_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 4h timeframe (no alignment needed as it's already 4h)
    upper_4h_4h = upper_4h
    lower_4h_4h = lower_4h
    
    # Volume filter: spike above 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Wait for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_4h[i]) or np.isnan(ema_200_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(upper_4h_4h[i]) or np.isnan(lower_4h_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long: price above 4h upper band, 1w uptrend (price > EMA200), 1d RSI not overbought (<70), volume breakout
            if (close[i] > upper_4h_4h[i] and 
                close[i] > ema_200_4h[i] and 
                rsi_1d_4h[i] < 70 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price below 4h lower band, 1w downtrend (price < EMA200), 1d RSI not oversold (>30), volume breakdown
            elif (close[i] < lower_4h_4h[i] and 
                  close[i] < ema_200_4h[i] and 
                  rsi_1d_4h[i] > 30 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 4h lower band or 1w trend reversal or RSI overbought
            if (close[i] < lower_4h_4h[i] or 
                close[i] < ema_200_4h[i] or 
                rsi_1d_4h[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 4h upper band or 1w trend reversal or RSI oversold
            if (close[i] > upper_4h_4h[i] or 
                close[i] > ema_200_4h[i] or 
                rsi_1d_4h[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals