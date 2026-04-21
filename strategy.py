#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2
Hypothesis: Daily Camarilla R1/S1 breakout with volume confirmation and 1-week EMA50 trend filter captures institutional momentum while avoiding whipsaw. Works in bull/bear markets by filtering breakouts with higher timeframe trend. Target 10-25 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.zeros_like(close_1w)
    ema50_1w[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's Camarilla levels
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_val = ph - pl
        if range_val > 0:
            R1[i] = pc + (range_val * 1.1 / 12)
            S1[i] = pc - (range_val * 1.1 / 12)
        else:
            R1[i] = pc
            S1[i] = pc
    
    # Volume filter: volume > 1.5x 20-day average
    volume = prices['volume'].values
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            volume_avg[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = R1[i]
        s1 = S1[i]
        ema50 = ema50_1w_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Calculate ATR for stoploss (20-day)
        if i >= 20:
            tr_values = []
            for j in range(1, 21):
                idx = i - j
                if idx >= 0:
                    tr = max(high[idx] - low[idx], 
                             abs(high[idx] - close[idx-1]), 
                             abs(low[idx] - close[idx-1]))
                    tr_values.append(tr)
            atr = np.mean(tr_values) if tr_values else 0
        else:
            atr = 0
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation in uptrend (price > weekly EMA50)
            if price > r1 and vol_confirm and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume confirmation in downtrend (price < weekly EMA50)
            elif price < s1 and vol_confirm and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to S1 or trend breaks
            if price < s1 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to R1 or trend breaks
            if price > r1 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2"
timeframe = "1d"
leverage = 1.0