#!/usr/bin/env python3
# 1d_cci_1w_trend_v1
# Hypothesis: Daily CCI with weekly trend filter for BTC/ETH/SOL. 
# Long when CCI crosses above -100 and price above weekly EMA200.
# Short when CCI crosses below +100 and price below weekly EMA200.
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 20-60 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_cci_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d CCI calculation (20-period)
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    tp_mad = np.where(tp_mad == 0, 1e-10, tp_mad)
    cci = (typical_price - tp_ma) / (0.015 * tp_mad)
    
    # 1w HTF data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if np.isnan(cci[i]) or np.isnan(ema_200_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 OR price below weekly EMA200
            if cci[i] < 100 or close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 OR price above weekly EMA200
            if cci[i] > -100 or close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: CCI crosses above -100 and price above weekly EMA200
            if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema_200_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: CCI crosses below +100 and price below weekly EMA200
            elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema_200_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals