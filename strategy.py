#!/usr/bin/env python3
"""
4h_rsi_pullback_1w_trend_volume_v1
Hypothesis: On 4-hour timeframe, buy pullbacks in weekly uptrend (price > weekly EMA200) when RSI(14) < 30 with volume > 1.5x average, sell rallies in weekly downtrend (price < weekly EMA200) when RSI(14) > 70 with volume > 1.5x average. Exit on opposite RSI extreme (RSI>70 for longs, RSI<30 for shorts). Uses weekly trend filter to avoid counter-trend trades, RSI for mean-reversion entries, and volume confirmation to ensure momentum. Designed for low frequency (~20-50 trades/year) to avoid fee drag while capturing mean-reversion within trends. Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend) by using weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1w_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_close = df_1w['close'].values
    w_ema200 = pd.Series(w_close).ewm(span=200, adjust=False).mean().values
    w_ema200_aligned = align_htf_to_ltf(prices, df_1w, w_ema200)
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if weekly EMA200 not available
        if np.isnan(w_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs weekly EMA200
        uptrend = close[i] > w_ema200_aligned[i]
        downtrend = close[i] < w_ema200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when RSI > 70 (overbought)
            if rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when RSI < 30 (oversold)
            if rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price pulls back to RSI < 30 in uptrend with volume confirmation
            long_entry = (rsi[i] < 30) and uptrend and vol_confirm
            # Short entry: price rallies to RSI > 70 in downtrend with volume confirmation
            short_entry = (rsi[i] > 70) and downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals