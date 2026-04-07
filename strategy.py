#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_trend_volume_v1
Hypothesis: On 4-hour timeframe, use RSI pullbacks during higher timeframe trends. Enter long when RSI < 30 in uptrend (1d EMA50) with volume > 1.5x average, short when RSI > 70 in downtrend. Exit when RSI crosses 50 in opposite direction. Designed for low frequency (20-50 trades/year) by requiring strong trend alignment and volume confirmation. Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend) by using 1-day trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    d_ema50 = pd.Series(d_close).ewm(span=50, adjust=False).mean().values
    d_ema50_aligned = align_htf_to_ltf(prices, df_1d, d_ema50)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if daily EMA50 not available
        if np.isnan(d_ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA50
        uptrend = close[i] > d_ema50_aligned[i]
        downtrend = close[i] < d_ema50_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when RSI crosses above 50
            if rsi[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when RSI crosses below 50
            if rsi[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30 in uptrend with volume confirmation
            long_entry = (rsi[i] < 30) and uptrend and vol_confirm
            # Short entry: RSI > 70 in downtrend with volume confirmation
            short_entry = (rsi[i] > 70) and downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals