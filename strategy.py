#!/usr/bin/env python3
"""
1d_sma_200_rsi_1w_trend_volume_v1
Hypothesis: Long when price > SMA200 and RSI(14) > 50 on 1d, with volume > 1.5x 20-day average. Short when price < SMA200 and RSI(14) < 50 on 1d, with volume > 1.5x 20-day average. Uses 1w trend filter (price > SMA50 on 1w) to avoid counter-trend trades. Works in bull/bear by following higher timeframe trend. Target: 10-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_sma_200_rsi_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d SMA200 for trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # 1d RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)  # handle division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w SMA50 for trend filter
    sma_50_1w = df_1w['close'].rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume confirmation (20-day average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(sma_200[i]) or np.isnan(rsi[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price < SMA200 or RSI < 40 or 1w trend turns bearish
            if close[i] < sma_200[i] or rsi[i] < 40 or close[i] < sma_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price > SMA200 or RSI > 60 or 1w trend turns bullish
            if close[i] > sma_200[i] or rsi[i] > 60 or close[i] > sma_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > SMA200 and RSI > 50 and 1w bullish trend with volume
            if (close[i] > sma_200[i] and rsi[i] > 50 and 
                close[i] > sma_50_1w_aligned[i] and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: price < SMA200 and RSI < 50 and 1w bearish trend with volume
            elif (close[i] < sma_200[i] and rsi[i] < 50 and 
                  close[i] < sma_50_1w_aligned[i] and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals