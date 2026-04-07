#!/usr/bin/env python3
"""
1h Momentum Reversal with 4h Trend Filter and Volume Confirmation
Long when 1h RSI < 30 and price bounces from support with rising volume AND 4h EMA trend up
Short when 1h RSI > 70 and price rejects resistance with rising volume AND 4h EMA trend down
Exit when RSI crosses 50
Uses mean reversion in strong trends to capture swing points with low frequency.
Target: 15-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_reversal_4h_trend_volume_v1"
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
    
    # === RSI (14) for mean reversion signals ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h EMA trend filter (21-period) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need rising volume (above average)
            if vol_ratio[i] < 1.1:
                signals[i] = 0.0
                continue
            
            # Entry: RSI extreme with volume confirmation AND 4h trend filter
            if rsi[i] < 30 and close[i] > close[i-1] and ema_4h_aligned[i] > ema_4h_aligned[i-1]:
                # Oversold bounce with rising volume in uptrend -> long
                position = 1
                signals[i] = 0.20
            elif rsi[i] > 70 and close[i] < close[i-1] and ema_4h_aligned[i] < ema_4h_aligned[i-1]:
                # Overbought rejection with rising volume in downtrend -> short
                position = -1
                signals[i] = -0.20
    
    return signals