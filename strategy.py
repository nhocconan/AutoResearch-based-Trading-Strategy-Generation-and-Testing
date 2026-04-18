#!/usr/bin/env python3
"""
4h_PriceAction_Trend_Momentum_v1
Hypothesis: In 4h timeframe, combine price action (Higher Highs/Lower Lows) with momentum (RSI) and trend (EMA) for high-probability entries.
Long when: Higher High + Higher Low pattern, RSI > 50, price > EMA(50).
Short when: Lower High + Lower Low pattern, RSI < 50, price < EMA(50).
Use volume confirmation to filter weak breakouts.
Target: 20-40 trades/year by requiring multiple confluence factors.
Works in both bull (trend continuation) and bear (trend reversal) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # Wilder smoothing for RSI
    for i in range(1, len(gain)):
        if i < 14:
            avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else np.nan
            avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else np.nan
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    vol_confirm = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Price action patterns (need at least 2 bars back)
        if i >= 2:
            hh_hl = (high[i] > high[i-1]) and (low[i] > low[i-1])  # Higher High, Higher Low
            lh_ll = (high[i] < high[i-1]) and (low[i] < low[i-1])  # Lower High, Lower Low
        else:
            hh_hl = False
            lh_ll = False
        
        if position == 0:
            # Long: Higher High + Higher Low, RSI > 50, price > EMA(50), volume confirmation
            if hh_hl and rsi[i] > 50 and close[i] > ema_50[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lower High + Lower Low, RSI < 50, price < EMA(50), volume confirmation
            elif lh_ll and rsi[i] < 50 and close[i] < ema_50[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Lower High + Lower Low OR RSI < 40
            if lh_ll or rsi[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Higher High + Higher Low OR RSI > 60
            if hh_hl or rsi[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceAction_Trend_Momentum_v1"
timeframe = "4h"
leverage = 1.0