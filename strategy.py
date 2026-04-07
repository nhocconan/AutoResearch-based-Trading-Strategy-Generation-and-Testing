#!/usr/bin/env python3
"""
1h_rsi_reversal_4h1d_volume_v1
Hypothesis: On 1-hour timeframe, use RSI mean reversion with 4h/1d trend filter and volume confirmation.
Long when RSI < 30 with price above 4h EMA(50) and 1d EMA(200) and volume > 1.5x 20-period average.
Short when RSI > 70 with price below 4h EMA(50) and 1d EMA(200) and volume > 1.5x 20-period average.
Exit when RSI returns to neutral zone (40-60).
Uses 4h/1d for trend direction, 1h only for entry timing. Designed for 15-35 trades/year to minimize fee drag.
Works in both bull/bear markets as RSI extremes occur in all regimes and volume filter ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_reversal_4h1d_volume_v1"
timeframe = "1h"
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
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: 20-period average on 1h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(14, 50, 200, 20), n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral zone (>= 40)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral zone (<= 60)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: RSI oversold with price above both EMAs
                if (rsi[i] < 30 and close[i] > ema_50_4h_aligned[i] and 
                    close[i] > ema_200_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: RSI overbought with price below both EMAs
                elif (rsi[i] > 70 and close[i] < ema_50_4h_aligned[i] and 
                      close[i] < ema_200_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals