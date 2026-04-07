#!/usr/bin/env python3
"""
12h_rsi_pullback_1w_ema_trend_v1
Hypothesis: On 12h timeframe, RSI pullbacks in the direction of weekly EMA trend provide high-probability entries.
In bull markets: buy RSI < 40 pullbacks above weekly EMA. In bear markets: sell RSI > 60 pullbacks below weekly EMA.
Uses volume confirmation to filter false signals and cooldown to prevent overtrading.
Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull/bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_pullback_1w_ema_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20)
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # RSI(14) on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Cooldown to prevent overtrading
    cooldown = 0
    cooldown_period = 4  # 4 bars = 48 hours minimum between trades
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1
        
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 60 (overbought) or trend change
            if rsi[i] > 60 or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
                cooldown = cooldown_period
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 (oversold) or trend change
            if rsi[i] < 40 or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
                cooldown = cooldown_period
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry (only if cooldown is 0)
            if cooldown > 0:
                signals[i] = 0.0
                continue
                
            # Long: RSI pullback (<40) in uptrend (price > weekly EMA) with volume
            if rsi[i] < 40 and close[i] > ema_20_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
                cooldown = cooldown_period
            # Short: RSI pullback (>60) in downtrend (price < weekly EMA) with volume
            elif rsi[i] > 60 and close[i] < ema_20_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
                cooldown = cooldown_period
    
    return signals