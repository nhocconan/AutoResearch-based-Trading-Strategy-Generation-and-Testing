#!/usr/bin/env python3
"""
1d_KAMA_Trend_with_RSI_Filter_and_ATR_Stop
Hypothesis: KAMA trend on daily timeframe combined with RSI filter and ATR-based stop.
Trades in direction of adaptive trend only when RSI confirms momentum.
ATR stop limits downside in volatile markets.
Designed for low trade frequency (<25/year) to minimize fee drag while capturing sustained trends.
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
    
    # Daily KAMA (adaptive trend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to daily
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Daily ATR(14) for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ATR to daily
    atr_aligned = align_htf_to_ltf(prices, df_1d, tr)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
            
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        atr_val = atr_aligned[i]
        
        if position == 0:
            # Long: price above KAMA AND RSI > 50 (bullish momentum)
            if price > kama_val and rsi_val > 50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below KAMA AND RSI < 50 (bearish momentum)
            elif price < kama_val and rsi_val < 50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                
        elif position == 1:
            signals[i] = 0.25
            # Exit: ATR-based stop or trend reversal
            if price <= entry_price - 1.5 * atr_val or price < kama_val:
                signals[i] = 0.0
                position = 0
                
        elif position == -1:
            signals[i] = -0.25
            # Exit: ATR-based stop or trend reversal
            if price >= entry_price + 1.5 * atr_val or price > kama_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_with_RSI_Filter_and_ATR_Stop"
timeframe = "1d"
leverage = 1.0