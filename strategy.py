#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Overbought_Oversold
Hypothesis: KAMA (Kaufman Adaptive Moving Average) defines trend direction, RSI identifies
overbought/oversold conditions for mean-reversion entries within the trend. Uses volume
confirmation to filter false signals. Designed to work in both bull (trend-following) and
bear (mean-reversion in downtrend) markets by aligning with higher timeframe trend.
Target: 20-30 trades/year to minimize fee drag while capturing high-probability mean-reversion
setups within the dominant trend.
"""

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
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    def kama(close_prices, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = np.full_like(close_prices, np.nan, dtype=float)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # RSI (Relative Strength Index)
    def rsi(close_prices, length=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close, length=14)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Higher timeframe trend filter: 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_vals[i]
        rsi_val = rsi_vals[i]
        ema50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI oversold (<30) + volume spike
            if price > kama_val and rsi_val < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI overbought (>70) + volume spike
            elif price < kama_val and rsi_val > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI overbought (>70) or price < KAMA (trend change)
            if rsi_val > 70 or price < kama_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI oversold (<30) or price > KAMA (trend change)
            if rsi_val < 30 or price > kama_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_RSI_Overbought_Oversold"
timeframe = "4h"
leverage = 1.0