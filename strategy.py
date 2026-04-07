#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_trend_volume_v1
Hypothesis: RSI pullbacks on 4h filtered by 1-day EMA200 trend and volume confirmation.
Long when RSI(14) < 30 (oversold) and price closes above EMA200(1d) with volume > average.
Short when RSI(14) > 70 (overbought) and price closes below EMA200(1d) with volume > average.
Designed for 15-25 trades/year on 4h with high-probability mean-reversion entries.
Works in bull markets (buy dips) and bear markets (sell rallies).
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
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(14) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # 1d trend filter
        above_1d_ema200 = close[i] > ema200_1d_aligned[i]
        below_1d_ema200 = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral or trend turns bearish
            if rsi[i] >= 50 or below_1d_ema200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral or trend turns bullish
            if rsi[i] <= 50 or above_1d_ema200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI oversold with volume confirmation and bullish trend
            if rsi_oversold and vol_confirmed and above_1d_ema200:
                position = 1
                signals[i] = 0.25
            # Short: RSI overbought with volume confirmation and bearish trend
            elif rsi_overbought and vol_confirmed and below_1d_ema200:
                position = -1
                signals[i] = -0.25
    
    return signals