#!/usr/bin/env python3
"""
4h_KAMA_Trend_Donchian20_VolumeBreakout
Hypothesis: KAMA adapts to market noise, identifying true trend state on 4h. 
Breakouts above/below Donchian(20) channels in direction of KAMA trend with 
volume confirmation (>1.3x average) capture momentum moves. 
In bull markets: longs on upside breakouts with uptrend KAMA.
In bear markets: shorts on downside breakouts with downtrend KAMA.
Volume filter reduces false breakouts. ATR-based stoploss manages risk.
Target: 20-40 trades/year (~80-160 over 4 years) to minimize fee drag.
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
    
    # Get 4h data for HTF trend filter (KAMA)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate KAMA on 4h close
    close_4h = df_4h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_4h, n=10))
    volatility = np.sum(np.abs(np.diff(close_4h, n=1)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA calculation
    kama = np.full_like(close_4h, np.nan, dtype=float)
    kama[9] = close_4h[9]  # Start after 10 periods
    for i in range(10, len(close_4h)):
        kama[i] = kama[i-1] + sc[i-1] * (close_4h[i] - kama[i-1])
    
    # Align KAMA to 15m timeframe (using 4h->15m: 4*15=60min per 4h bar)
    # Since we're on 4h primary timeframe, alignment is 1:1 but need to ensure
    # we use completed 4h bar only
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Donchian(20) channels on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_avg)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10), Donchian (20), volume avg (20), ATR (14)
    start_idx = max(10, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        donch_high = high_max[i]
        donch_low = low_min[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Determine trend: price > KAMA = uptrend, price < KAMA = downtrend
            is_uptrend = close_val > kama_val
            is_downtrend = close_val < kama_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above Donchian high and volume confirms
                if (close_val > donch_high) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below Donchian low and volume confirms
                if (close_val < donch_low) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches Donchian low or trend changes to downtrend
            # Or ATR-based stop: price < highest high since entry - 2*ATR
            exit_condition = (close_val < donch_low) or (close_val < kama_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches Donchian high or trend changes to uptrend
            # Or ATR-based stop: price > lowest low since entry + 2*ATR
            exit_condition = (close_val > donch_high) or (close_val > kama_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_Donchian20_VolumeBreakout"
timeframe = "4h"
leverage = 1.0