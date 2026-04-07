#!/usr/bin/env python3
"""
1d_weekly_rsi_momentum_v1
Hypothesis: Weekly RSI with momentum confirmation on 1d timeframe captures intermediate-term trend reversals with high accuracy. 
Enter long when weekly RSI crosses above 30 with bullish daily momentum (close > daily SMA20), 
enter short when weekly RSI crosses below 70 with bearish daily momentum (close < daily SMA20). 
Weekly timeframe filters out short-term noise and prevents counter-trend trades. 
Target: 10-20 trades/year on 1d timeframe to minimize fee drag while maintaining signal quality.
Works in both bull and bear markets by following weekly momentum extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI(14) calculation
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14w = 100 - (100 / (1 + rs))
    
    # Daily SMA20 for momentum confirmation
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly RSI to 1d timeframe
    rsi_14w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(rsi_14w_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(sma_20[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        rsi_weekly = rsi_14w_aligned[i]
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: weekly RSI falls below 50 or momentum turns bearish
            if rsi_weekly < 50 or close[i] < sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly RSI rises above 50 or momentum turns bullish
            if rsi_weekly > 50 or close[i] > sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: weekly RSI crosses above 30 with bullish momentum and volume
            if i > 50 and rsi_14w_aligned[i-1] <= 30 and rsi_weekly > 30 and close[i] > sma_20[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: weekly RSI crosses below 70 with bearish momentum and volume
            elif i > 50 and rsi_14w_aligned[i-1] >= 70 and rsi_weekly < 70 and close[i] < sma_20[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals