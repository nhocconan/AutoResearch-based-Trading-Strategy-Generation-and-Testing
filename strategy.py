#!/usr/bin/env python3
# 1d_1w_kama_rsi_volume_v1
# Strategy: 1-day KAMA direction with 1-week trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets. Combined with 1-week trend alignment and volume confirmation, it captures strong trends while avoiding whipsaws. Designed for low trade frequency (10-30/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 1d data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 1d (it's already 1d, but for consistency)
    kama_aligned = kama  # no alignment needed as both are 1d
    
    # Calculate RSI on 1d
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # 1-week EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = np.zeros(n)
    for i in range(20, n):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # KAMA direction: price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # 1-week EMA trend filter
        trend_bullish = close[i] > ema_20_1w_aligned[i]
        trend_bearish = close[i] < ema_20_1w_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_not_extreme = (rsi[i] > 20) and (rsi[i] < 80)
        
        # Entry conditions
        # Long: Price above KAMA AND bullish trend AND volume confirmation AND RSI not extreme
        if price_above_kama and trend_bullish and vol_confirm and rsi_not_extreme and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price below KAMA AND bearish trend AND volume confirmation AND RSI not extreme
        elif price_below_kama and trend_bearish and vol_confirm and rsi_not_extreme and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite KAMA cross
        elif position == 1 and price_below_kama:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_above_kama:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals