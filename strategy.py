#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d ATR-based volatility filter and 12h RSI mean reversion.
# Long: RSI(14) < 30 AND ATR(14) > 1.5x ATR(50) (high volatility oversold).
# Short: RSI(14) > 70 AND ATR(14) > 1.5x ATR(50) (high volatility overbought).
# Exit: RSI crosses back to neutral (40-60 range).
# Uses volatility filter to trade mean reversion only during high vol regimes, avoiding chop.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    atr_14 = np.zeros(n)
    atr_50 = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
        
        if i < 50:
            atr_50[i] = np.nan
        elif i == 50:
            atr_50[i] = np.mean(tr[1:51])
        else:
            atr_50[i] = (atr_50[i-1] * 49 + tr[i]) / 50
    
    # Volatility filter: ATR(14) > 1.5 * ATR(50)
    vol_filter = (atr_14 > 1.5 * atr_50) & (~np.isnan(atr_14)) & (~np.isnan(atr_50))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        if np.isnan(rsi[i]) or not vol_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold in high volatility
            if rsi[i] < 30:
                position = 1
                signals[i] = position_size
            # Short: RSI overbought in high volatility
            elif rsi[i] > 70:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (>= 40)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (<= 60)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_RSI_Volatility_MeanReversion"
timeframe = "12h"
leverage = 1.0