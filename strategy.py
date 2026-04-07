#!/usr/bin/env python3
"""
6h_cci_rsi_confluence_1d_trend_v1
Hypothesis: On 6-hour timeframe, combine CCI(20) for momentum extremes with RSI(14) for overbought/oversold conditions, filtered by 1-day EMA(50) trend direction. Long when CCI < -100 and RSI < 30 with price above daily EMA(50). Short when CCI > 100 and RSI > 70 with price below daily EMA(50). Exit when CCI crosses back toward zero (|CCI| < 20). Designed for low-frequency, high-conviction trades (15-35/year) to minimize fee drag while capturing mean-reversion in overextended moves during trends. Works in both bull and bear markets as CCI adapts to volatility and trend filter ensures alignment with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_rsi_confluence_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate CCI(20) on 6h data
    typical_price = (high + low + close) / 3
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - tp_mean) / (0.015 * tp_mad)
    cci = cci.values  # convert to numpy array
    
    # Calculate RSI(14) on 6h data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values  # convert to numpy array
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 14, 50), n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        if position == 1:  # Long position
            # Exit: CCI crosses back toward zero (|CCI| < 20)
            if abs(cci[i]) < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses back toward zero (|CCI| < 20)
            if abs(cci[i]) < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: CCI < -100 (oversold) AND RSI < 30 (oversold) AND price above daily EMA(50)
            if (cci[i] < -100 and rsi[i] < 30 and close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: CCI > 100 (overbought) AND RSI > 70 (overbought) AND price below daily EMA(50)
            elif (cci[i] > 100 and rsi[i] > 70 and close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals