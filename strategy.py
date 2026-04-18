#!/usr/bin/env python3
"""
4h_Volume_Weighted_RSI_Divergence_With_Volatility_Filter
Hypothesis: Identify momentum exhaustion via RSI divergence confirmed by volume imbalance and volatility contraction. Enter long on bullish divergence with expanding volume, short on bearish divergence with contracting volume. Uses RSI(14) for momentum, volume-weighted price for confirmation, and ATR-based volatility filter to avoid chop. Designed for low frequency (<30 trades/year) with high win rate in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume-weighted price (VWAP approximation for divergence)
    typical_price = (high + low + close) / 3
    vwp_num = np.cumsum(typical_price * volume)
    vwp_den = np.cumsum(volume)
    vwp = np.where(vwp_den != 0, vwp_num / vwp_den, typical_price)
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])
    for i in range(15, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Divergence detection: look for price making new high/low while RSI does not
    lookback = 10
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if low[i] == np.min(low[i-lookback:i+1]) and rsi[i] > np.min(rsi[i-lookback:i+1]):
            # Find if there's a lower low in price within lookback
            if np.any(low[i-lookback:i] < low[i]):
                bullish_div[i] = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if high[i] == np.max(high[i-lookback:i+1]) and rsi[i] < np.max(rsi[i-lookback:i+1]):
            # Find if there's a higher high in price within lookback
            if np.any(high[i-lookback:i] > high[i]):
                bearish_div[i] = True
    
    # Volume confirmation: increasing volume on bullish divergence, decreasing on bearish
    vol_ma = np.zeros_like(volume)
    vol_ma[19] = np.mean(volume[0:20])
    for i in range(20, len(volume)):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    vol_increasing = volume > vol_ma
    vol_decreasing = volume < vol_ma
    
    # Volatility filter: avoid trading when ATR is too high (choppy markets)
    atr_ma = np.zeros_like(atr)
    atr_ma[19] = np.mean(atr[0:20])
    for i in range(20, len(atr)):
        atr_ma[i] = (atr_ma[i-1] * 19 + atr[i]) / 20
    
    vol_filter = atr < (atr_ma * 1.5)  # Only trade when volatility is below 1.5x MA
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(vwp[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_increasing[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: bullish divergence + increasing volume + low volatility
            if bullish_div[i] and vol_increasing[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + decreasing volume + low volatility
            elif bearish_div[i] and vol_decreasing[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: bearish divergence or volatility expansion
            if bearish_div[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: bullish divergence or volatility expansion
            if bullish_div[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Volume_Weighted_RSI_Divergence_With_Volatility_Filter"
timeframe = "4h"
leverage = 1.0