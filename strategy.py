#!/usr/bin/env python3
"""
6h_RSI2_Stochastic_MeanReversion_12hTrend
Hypothesis: Mean reversion on extreme RSI(2) with Stochastic oversold/overbought and 12h trend filter captures reversals in both bull and bear markets. 
Long when RSI(2) < 10, Stochastic %K < 20, and 12h trend up; short when RSI(2) > 90, Stochastic %K > 80, and 12h trend down. 
Targets 12-37 trades/year on 6h to minimize fee drag while capturing mean-reversion opportunities.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # RSI(2) on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Stochastic(14,3,3) on 6h data
    low_min = pd.Series(low).rolling(window=14, min_periods=14).min().values
    high_max = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_percent_k = np.divide((close - low_min), (high_max - low_min), out=np.zeros_like(close), where=(high_max - low_min)!=0) * 100
    stoch_percent_k_smooth = pd.Series(stoch_percent_k).rolling(window=3, min_periods=3).mean().values
    stoch_percent_d = pd.Series(stoch_percent_k_smooth).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = max(50, 14 + 3 + 3)  # 12h EMA50 + Stochastic lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(rsi2[i]) or 
            np.isnan(stoch_percent_k_smooth[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi2[i]
        stoch_k = stoch_percent_k_smooth[i]
        ema_trend = ema50_12h_aligned[i]
        
        if position == 0:
            # Long: RSI(2) oversold, Stochastic oversold, and 12h uptrend
            if rsi_val < 10 and stoch_k < 20 and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI(2) overbought, Stochastic overbought, and 12h downtrend
            elif rsi_val > 90 and stoch_k > 80 and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI(2) overbought or trend turns down
            if rsi_val > 70 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI(2) oversold or trend turns up
            if rsi_val < 30 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI2_Stochastic_MeanReversion_12hTrend"
timeframe = "6h"
leverage = 1.0