#!/usr/bin/env python3
"""
1h_momentum_4h1d_volume_v1
Hypothesis: In trending markets (4h/1d aligned), 1h momentum with volume confirmation captures trend continuation. 
Uses 4h EMA for trend direction, 1h RSI for momentum entry, and volume spike for confirmation. 
Designed for low trade frequency (15-37/year) to minimize fee drag. Works in bull/bear by only trading with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    ema40_4h = pd.Series(df_4h['close'].values).ewm(span=40, adjust=False).mean().values
    ema40_4h_aligned = align_htf_to_ltf(prices, df_4h, ema40_4h)
    
    # 1d EMA for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1h volume spike filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema40_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: trend turns down OR RSI overbought
            if close[i] < ema40_4h_aligned[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: trend turns up OR RSI oversold
            if close[i] > ema40_4h_aligned[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price above 4h/1d EMA + RSI momentum + volume
            if (close[i] > ema40_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                rsi[i] > 55 and rsi[i] < 70 and 
                vol_confirm):
                position = 1
                signals[i] = 0.20
            # Short: price below 4h/1d EMA + RSI momentum + volume
            elif (close[i] < ema40_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  rsi[i] < 45 and rsi[i] > 30 and 
                  vol_confirm):
                position = -1
                signals[i] = -0.20
    
    return signals