#!/usr/bin/env python3
"""
1h_rsi_pullback_4h1d_volume_v1
Hypothesis: RSI pullbacks (RSI<30 for long, RSI>70 for short) combined with 4h/1d trend filters and volume confirmation on 1h timeframe. Uses 4h EMA50 and 1d EMA200 for trend direction. Volume filter requires current volume > 1.5x 20-period average. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1d_volume_v1"
timeframe = "1h"
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
    
    # 4h data for intermediate trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_4h_50 = df_4h['close'].ewm(span=50, adjust=False).mean()
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50.values)
    
    # 1d data for long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema_1d_200 = df_1d['close'].ewm(span=200, adjust=False).mean()
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200.values)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral RSI when no loss
    
    # Volume confirmation (20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_50_aligned[i]) or 
            np.isnan(ema_1d_200_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI rises above 70 or price breaks below 4h EMA50 or 1d EMA200
            if (rsi[i] > 70 or close[i] < ema_4h_50_aligned[i] or 
                close[i] < ema_1d_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI falls below 30 or price breaks above 4h EMA50 or 1d EMA200
            if (rsi[i] < 30 or close[i] > ema_4h_50_aligned[i] or 
                close[i] > ema_1d_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold), with volume and price above both EMAs
            if (rsi[i] < 30 and vol_confirm and 
                close[i] > ema_4h_50_aligned[i] and 
                close[i] > ema_1d_200_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 (overbought), with volume and price below both EMAs
            elif (rsi[i] > 70 and vol_confirm and 
                  close[i] < ema_4h_50_aligned[i] and 
                  close[i] < ema_1d_200_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals