#!/usr/bin/env python3
"""
1h Momentum Reversal with 4h/1d Trend Filter
Hypothesis: In strong trends (4h/1d aligned), 1h pullbacks offer high-probability entries.
In bull markets: buy pullbacks in uptrends. In bear markets: sell rallies in downtrends.
Uses RSI(2) for extreme pullbacks, volume surge for confirmation, and 4h/1d EMA alignment for trend.
Target: 15-30 trades/year by requiring multiple confluence factors.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_reversal_4h1d_trend_v1"
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
    
    # === RSI(2) for extreme pullbacks ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume surge confirmation ===
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 4h trend filter (EMA 50) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d trend filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(rsi[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 70 (overbought) or trend weakens
            if rsi[i] > 70 or ema_4h_aligned[i] < ema_4h_aligned[i-1] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 30 (oversold) or trend weakens
            if rsi[i] < 30 or ema_4h_aligned[i] > ema_4h_aligned[i-1] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume surge (above 1.5x average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: RSI extreme pullback with volume surge AND 4h/1d trend alignment
            if rsi[i] < 15 and ema_4h_aligned[i] > ema_4h_aligned[i-1] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Extreme oversold in uptrend -> long
                position = 1
                signals[i] = 0.20
            elif rsi[i] > 85 and ema_4h_aligned[i] < ema_4h_aligned[i-1] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Extreme overbought in downtrend -> short
                position = -1
                signals[i] = -0.20
    
    return signals