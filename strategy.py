#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour Supertrend for trend direction and 1-hour RSI for entry timing.
# Uses Supertrend (ATR-based) to identify trend on 4h (bull/bear) and only takes entries in trend direction.
# RSI(14) on 1h: long when RSI < 30 (oversold) in uptrend, short when RSI > 70 (overbought) in downtrend.
# Includes session filter (08:00-20:00 UTC) to avoid low-liquidity hours.
# Fixed position size of 0.20 to control risk and reduce overtrading.
# Designed for 1h timeframe targeting 60-150 total trades over 4 years.

name = "1h_supertrend4h_rsi14_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h Supertrend for trend direction
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) on 4h
    atr_period = 10
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h[:-1]),
            np.abs(low_4h[1:] - close_4h[:-1])
        )
    )
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = np.full(len(close_4h), np.nan)
    for i in range(atr_period, len(close_4h)):
        if i == atr_period:
            atr_4h[i] = np.nanmean(tr_4h[i-atr_period+1:i+1])
        else:
            atr_4h[i] = (atr_4h[i-1] * (atr_period - 1) + tr_4h[i]) / atr_period
    
    # Supertrend calculation
    factor = 3.0
    upper_band_4h = (high_4h + low_4h) / 2 + factor * atr_4h
    lower_band_4h = (high_4h + low_4h) / 2 - factor * atr_4h
    
    supertrend_4h = np.full(len(close_4h), np.nan)
    uptrend_4h = np.full(len(close_4h), True)
    
    for i in range(1, len(close_4h)):
        if np.isnan(upper_band_4h[i-1]) or np.isnan(lower_band_4h[i-1]):
            continue
            
        if close_4h[i] > upper_band_4h[i-1]:
            uptrend_4h[i] = True
        elif close_4h[i] < lower_band_4h[i-1]:
            uptrend_4h[i] = False
        else:
            uptrend_4h[i] = uptrend_4h[i-1]
            if uptrend_4h[i] and lower_band_4h[i] < lower_band_4h[i-1]:
                lower_band_4h[i] = lower_band_4h[i-1]
            if not uptrend_4h[i] and upper_band_4h[i] > upper_band_4h[i-1]:
                upper_band_4h[i] = upper_band_4h[i-1]
        
        if uptrend_4h[i]:
            supertrend_4h[i] = lower_band_4h[i]
        else:
            supertrend_4h[i] = upper_band_4h[i]
    
    # Align Supertrend to 1h
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float)) > 0.5
    
    # 1h RSI(14)
    rsi_period = 14
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    for i in range(rsi_period, len(close)):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(rsi[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI > 50 (momentum fade) or price below Supertrend
            if rsi[i] > 50 or close[i] < supertrend_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 50 (momentum fade) or price above Supertrend
            if rsi[i] < 50 or close[i] > supertrend_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of 4h trend
            if uptrend_4h_aligned[i]:
                # Long: RSI < 30 (oversold) in uptrend
                if rsi[i] < 30:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
            else:
                # Short: RSI > 70 (overbought) in downtrend
                if rsi[i] > 70:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals