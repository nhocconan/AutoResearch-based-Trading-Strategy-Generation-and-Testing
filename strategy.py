#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter_V2
Hypothesis: Use 1-day KAMA to capture primary trend direction and RSI(14) for momentum filter. 
Add 1-week EMA for higher timeframe trend confirmation. Only take trades when KAMA direction 
aligns with weekly EMA and RSI is not extreme. Designed for very low trade frequency (<15/year) 
to minimize fee decay while capturing major trends in both bull and bear markets.
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
    
    # Calculate Kaufman Adaptive Moving Average (KAMA)
    # Parameters: ER length=10, Fast=2, Slow=30
    er_len = 10
    fast_sc = 2
    slow_sc = 30
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_len, n):
        if i >= er_len:
            net_change = np.abs(close[i] - close[i-er_len])
            total_change = np.sum(volatility[i-er_len+1:i+1])
            if total_change > 0:
                er[i] = net_change / total_change
            else:
                er[i] = 0
    
    # Smoothing Constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-week EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 21)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema_1w_val = ema_1w_aligned[i]
        rsi_val = rsi[i]
        vol_ok = volume_ok[i]
        
        if position == 0:
            # Long: price above KAMA and weekly EMA, RSI not overbought, volume ok
            if price > kama_val and price > ema_1w_val and rsi_val < 70 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and weekly EMA, RSI not oversold, volume ok
            elif price < kama_val and price < ema_1w_val and rsi_val > 30 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below KAMA or weekly EMA
            if price < kama_val or price < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above KAMA or weekly EMA
            if price > kama_val or price > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter_V2"
timeframe = "1d"
leverage = 1.0