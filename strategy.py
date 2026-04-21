#!/usr/bin/env python3
"""
1d_1w_Keltner_Breakout_VolumeRegime
Hypothesis: Keltner Channel breakouts on 1d with 1w trend filter (price above/below 1w EMA20) and volume confirmation (>1.3x 20-day average) capture strong trending moves while avoiding sideways chop. Designed for low trade frequency (target: 15-25/year) to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn. Works in bull markets via upward breakouts and bear markets via downward breakouts with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data once for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i < 20:
            ema20_1w[i] = np.mean(close_1w[:i+1])
        else:
            ema20_1w[i] = np.mean(close_1w[i-20+1:i+1])
    
    # Align 1w EMA20 to 1d timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 1.5) on 1d
    # Middle line: EMA20
    ema20 = np.zeros_like(close)
    for i in range(n):
        if i < 20:
            ema20[i] = np.mean(close[:i+1])
        else:
            ema20[i] = np.mean(close[i-20+1:i+1])
    
    # Average True Range (14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    # Keltner Bands
    upper = ema20 + 1.5 * atr
    lower = ema20 - 1.5 * atr
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1])
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema20_val = ema20[i]
        upper_val = upper[i]
        lower_val = lower[i]
        ema20_1w = ema20_1w_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.0 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner band with volume and 1w uptrend (price > 1w EMA20)
            if price > upper_val and vol_ok and price > ema20_1w:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Keltner band with volume and 1w downtrend (price < 1w EMA20)
            elif price < lower_val and vol_ok and price < ema20_1w:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below middle line (EMA20) or breaks below 1w EMA20 (trend change)
            if price < ema20_val or price < ema20_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above middle line (EMA20) or breaks above 1w EMA20 (trend change)
            if price > ema20_val or price > ema20_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Keltner_Breakout_VolumeRegime"
timeframe = "1d"
leverage = 1.0