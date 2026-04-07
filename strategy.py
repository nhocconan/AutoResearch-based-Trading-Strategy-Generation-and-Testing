#!/usr/bin/env python3
"""
4h_keltner_meanrev_volume_v1
Hypothesis: On 4h timeframe, enter long when price touches lower Keltner band with above-average volume and oversold RSI, enter short when price touches upper Keltner band with above-average volume and overbought RSI. Uses 1d EMA trend filter to avoid counter-trend trades. Keltner channels adapt to volatility, making them effective in both trending and ranging markets. Designed for 20-50 trades/year to minimize fee drift while capturing mean reversion at volatility extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_keltner_meanrev_volume_v1"
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
    
    # Calculate 20-period EMA for Keltner middle
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).values
    
    # Calculate ATR (10-period) for Keltner width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Keltner bands
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses back above EMA20 (mean reversion complete)
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back below EMA20 (mean reversion complete)
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price at lower Keltner band with oversold RSI and 1d EMA uptrend
                if close[i] <= keltner_lower[i] and rsi[i] < 30 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price at upper Keltner band with overbought RSI and 1d EMA downtrend
                elif close[i] >= keltner_upper[i] and rsi[i] > 70 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals