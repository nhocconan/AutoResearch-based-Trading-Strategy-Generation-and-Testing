#!/usr/bin/env python3
"""
Hypothesis: 6h momentum strategy using 12h EMA for trend direction and 6h RSI for mean-reversion entries.
Trades only when 12h EMA slope confirms trend and 6h RSI is oversold/overbought within that trend.
Designed to work in both bull and bear markets by using trend-following with pullback entries.
Target: 25-35 trades/year per symbol (100-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(34) on close
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h EMA slope (trend strength) - positive = uptrend, negative = downtrend
    ema_slope = np.diff(ema_12h_aligned, prepend=ema_12h_aligned[0])
    
    # 6h RSI(14) for mean-reversion entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        if position == 0:
            # Long: Uptrend (positive EMA slope) + RSI oversold + volume confirmation
            if ema_slope[i] > 0 and rsi[i] < 30 and vol > 1.2 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (negative EMA slope) + RSI overbought + volume confirmation
            elif ema_slope[i] < 0 and rsi[i] > 70 and vol > 1.2 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought or trend weakening
            if rsi[i] > 70 or ema_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI oversold or trend weakening
            if rsi[i] < 30 or ema_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA34_RSI_Volume"
timeframe = "6h"
leverage = 1.0