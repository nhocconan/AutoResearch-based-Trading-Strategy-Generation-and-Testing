#!/usr/bin/env python3
"""
1h_VWAP_Deviation_Trend_4h_1d
Hypothesis: In 1h timeframe, price tends to revert to 4h VWAP during ranging markets but breaks with trend in strong moves. 
Long when: price < 4h VWAP (oversold) AND 1d trend up (price > 1d EMA50) AND volume spike.
Short when: price > 4h VWAP (overbought) AND 1d trend down (price < 1d EMA50) AND volume spike.
Use 4h VWAP for mean reversion signal, 1d EMA50 for trend filter, volume spike for confirmation.
Targets 15-30 trades/year by requiring all three conditions.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

name = "1h_VWAP_Deviation_Trend_4h_1d"
timeframe = "1h"
leverage = 1.0

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
    
    # === 4H Data for VWAP ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Typical price and VWAP calculation
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_numerator = np.cumsum(typical_price_4h * volume_4h)
    vwap_denominator = np.cumsum(volume_4h)
    vwap_4h = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price_4h)
    
    # Align 4h VWAP to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.5x 24-period average (more selective)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below 4h VWAP (oversold) AND 1d uptrend AND volume spike
            if close[i] < vwap_4h_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price above 4h VWAP (overbought) AND 1d downtrend AND volume spike
            elif close[i] > vwap_4h_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back above VWAP (mean reversion) OR trend breaks
            if close[i] > vwap_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price crosses back below VWAP (mean reversion) OR trend breaks
            if close[i] < vwap_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals