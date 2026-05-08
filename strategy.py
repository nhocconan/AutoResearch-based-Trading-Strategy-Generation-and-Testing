#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Trend_RSI_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily KAMA for trend direction
    close_1d_series = pd.Series(close_1d)
    delta = close_1d_series.diff().abs()
    direction = abs(close_1d_series.diff(10))
    volatility = delta.rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close_1d[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc[i] * (close_1d[i] - kama[-1]))
    kama = np.array(kama)
    kama_1d = kama
    
    # Align KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Daily RSI for mean reversion filter
    delta_rsi = pd.Series(close_1d).diff()
    gain = delta_rsi.where(delta_rsi > 0, 0)
    loss = -delta_rsi.where(delta_rsi < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi_1d = rsi.values
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI oversold, volume spike
            long_cond = (close[i] > kama_1d_aligned[i] and 
                        rsi_1d_aligned[i] < 30 and
                        volume_spike[i])
            
            # Short: price below KAMA (downtrend), RSI overbought, volume spike
            short_cond = (close[i] < kama_1d_aligned[i] and 
                         rsi_1d_aligned[i] > 70 and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI overbought
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI oversold
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals