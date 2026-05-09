#!/usr/bin/env python3
# 4h_RSI_Pullback_With_Volume_Spice
# Hypothesis: RSI pullback strategy with volume spike confirmation and 4h EMA trend filter.
# Works in bull/bear: Trend filter ensures trades align with higher timeframe momentum, 
# RSI pullback (from overbought/oversold) provides mean reversion entries, volume spike confirms institutional interest.
# Uses 4h EMA50 for trend, RSI(14) for mean reversion signals, and volume ratio (>2.0) for confirmation.
# Designed for moderate trade frequency (~25-40 trades/year) to minimize fee drag.

name = "4h_RSI_Pullback_With_Volume_Spice"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else 0
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else 0
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = np.full_like(close, np.nan)
    if len(close) >= 50:
        ema_50[49] = np.mean(close[0:50])
        for i in range(50, len(close)):
            ema_50[i] = (ema_50[i-1] * 49 + close[i]) / 50
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(ema_50[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) AND price > EMA50 (uptrend) AND volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_50[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought) AND price < EMA50 (downtrend) AND volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_50[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) OR trend reversal (price < EMA50)
            if rsi[i] > 50 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) OR trend reversal (price > EMA50)
            if rsi[i] < 50 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals