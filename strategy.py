#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI_Tick_Filter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d RSI(14) - mean reversion signal
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align both to 6h
    rsi_6h = align_htf_to_ltf(prices, df_1d, rsi_values)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h RSI(2) for entry timing - short-term mean reversion
    rsi_fast = pd.Series(close).diff()
    gain_fast = rsi_fast.clip(lower=0)
    loss_fast = -rsi_fast.clip(upper=0)
    avg_gain_fast = gain_fast.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss_fast = loss_fast.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs_fast = avg_gain_fast / (avg_loss_fast + 1e-10)
    rsi_fast_values = 100 - (100 / (1 + rs_fast))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_6h[i]) or np.isnan(ema50_1d_6h[i]) or 
            np.isnan(rsi_fast_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_1d = rsi_6h[i]
        ema50 = ema50_1d_6h[i]
        rsi_fast = rsi_fast_values[i]
        
        if position == 0:
            # Long: 1d RSI oversold (<30) + 6s RSI extremely oversold (<10) + above 1d EMA50
            if rsi_1d < 30 and rsi_fast < 10 and close[i] > ema50:
                signals[i] = 0.25
                position = 1
            # Short: 1d RSI overbought (>70) + 6s RSI extremely overbought (>90) + below 1d EMA50
            elif rsi_1d > 70 and rsi_fast > 90 and close[i] < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d RSI overbought or price below EMA50
            if rsi_1d > 70 or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d RSI oversold or price above EMA50
            if rsi_1d < 30 or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals