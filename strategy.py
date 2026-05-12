#!/usr/bin/env python3
name = "1d_Stochastic_RSI_Bollinger_Reversal_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Stochastic RSI (14,14,3,3)
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    # Calculate RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Stochastic of RSI
    rsi_min = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).max().values
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    
    # %K and %D
    k = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean().values
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean().values
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, rsi_period + stoch_period + k_period + d_period, bb_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(k[i]) or np.isnan(d[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: StochRSI oversold + price near lower BB + weekly uptrend
            if (k[i] < 20 and d[i] < 20 and 
                close[i] <= lower[i] * 1.01 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: StochRSI overbought + price near upper BB + weekly downtrend
            elif (k[i] > 80 and d[i] > 80 and 
                  close[i] >= upper[i] * 0.99 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: StochRSI overbought or price touches upper BB
            if k[i] > 80 or close[i] >= upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: StochRSI oversold or price touches lower BB
            if k[i] < 20 or close[i] <= lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals