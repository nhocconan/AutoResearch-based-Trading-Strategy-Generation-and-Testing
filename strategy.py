#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Trend
# Hypothesis: KAMA(10) captures adaptive trend; RSI(14) filters overbought/oversold in trend direction. Works in bull (trend+pullback long) and bear (trend+pullback short). Volume surge confirms momentum. Targets 20-30 trades/year to minimize fee drag.

name = "4h_KAMA_Direction_RSI_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA(10) - adaptive trend
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (6-period = 1.5 days)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    # Warmup: KAMA/RSI need 14, volume MA needs 6
    start_idx = 14
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Volume surge (1.6x average)
        volume_surge = volume[i] > 1.6 * vol_ma[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: price > KAMA, RSI < 60 (not overbought), volume surge
            if close[i] > kama[i] and rsi[i] < 60 and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI > 40 (not oversold), volume surge
            elif close[i] < kama[i] and rsi[i] > 40 and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            if bars_since_entry < 2:
                signals[i] = signals[i-1]
                continue
            
            if position == 1:
                # Exit: price < KAMA or RSI > 70 (overbought)
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: price > KAMA or RSI < 30 (oversold)
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals