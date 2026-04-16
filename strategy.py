# 1d_RSI_Pullback_KAMA_Trend - Long-only strategy using KAMA trend filter and RSI pullback entries on daily timeframe
# Works in bull via trend-following and in bear via oversold bounces in downtrend (avoided by trend filter)
# Target: 10-25 trades/year with low frequency to minimize fee drag
# Uses 1-week trend filter for multi-timeframe alignment

#!/usr/bin/env python3
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
    
    # === 1d indicators (primary timeframe) ===
    # KAMA for trend direction
    def calculate_kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(change.shape) > 1 else \
                    np.convolve(np.abs(np.diff(close, prepend=close[0])), np.ones(er_length), 'same')
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # RSI(14) for pullback entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1w trend filter (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Simple weekly trend: price above 20-week EMA
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        weekly_ema = ema_20_1w_aligned[i]
        
        # Exit conditions
        if position == 1:
            # Exit when price closes below KAMA or RSI becomes overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # Long: Price above weekly EMA (uptrend filter) AND RSI pulled back to oversold
            if price > weekly_ema and rsi_val < 30:
                signals[i] = 0.25
                position = 1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_RSI_Pullback_KAMA_Trend"
timeframe = "1d"
leverage = 1.0