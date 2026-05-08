#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) divergence with 1d MACD trend filter and volume spike
# Uses RSI divergences for high-probability reversals in both bull and bear markets.
# Requires alignment with daily MACD histogram sign and volume spike to filter false signals.
# Designed for low-frequency trades (<100 total) to minimize fee drag.

name = "4h_RSI_Divergence_1dMACD_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for MACD trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily MACD (12,26,9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align MACD histogram to 4h timeframe
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike (2x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # RSI divergence detection
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    # Look for bullish divergence: price makes lower low, RSI makes higher low
    for i in range(14, n):
        if i < 28:  # Need at least 2 periods to compare
            continue
            
        # Check for price lower low
        if low[i] < low[i-14]:
            # Look for higher low in RSI over same period
            if np.any(rsi[i-14:i] > np.min(rsi[i-28:i-14])):
                bullish_div[i] = True
                
        # Check for bearish divergence: price makes higher high, RSI makes lower high
        if high[i] > high[i-14]:
            # Look for lower high in RSI over same period
            if np.any(rsi[i-14:i] < np.max(rsi[i-28:i-14])):
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure MACD has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(macd_hist_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish RSI divergence with 1d bullish MACD and volume spike
            if (bullish_div[i] and 
                macd_hist_aligned[i] > 0 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish RSI divergence with 1d bearish MACD and volume spike
            elif (bearish_div[i] and 
                  macd_hist_aligned[i] < 0 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish RSI divergence or MACD turns bearish
            if (bearish_div[i] or 
                macd_hist_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish RSI divergence or MACD turns bullish
            if (bullish_div[i] or 
                macd_hist_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals