#!/usr/bin/env python3
"""
1h_4h1d_CCI_MeanReversion_WithVolume
Hypothesis: On 1h timeframe, use daily CCI to detect oversold/overbought conditions (CCI < -100 for long, CCI > +100 for short)
and 4h EMA for trend filter (only long when price > EMA, short when price < EMA).
Require volume spike (>1.5x 20-period average) to confirm entry.
Trade only during active session (08-20 UTC).
Target: 15-35 trades/year per symbol, using mean reversion in ranging markets and trend alignment in trending markets.
"""

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
    
    # Get daily data for CCI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CCI(20) on daily
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = np.full(len(typical_price), np.nan)
    mad = np.full(len(typical_price), np.nan)
    cci = np.full(len(typical_price), np.nan)
    
    if len(typical_price) >= 20:
        for i in range(19, len(typical_price)):
            sma_tp[i] = np.mean(typical_price[i-19:i+1])
            mad[i] = np.mean(np.abs(typical_price[i-19:i+1] - sma_tp[i]))
            if mad[i] != 0:
                cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mad[i])
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 20:
        ema_4h[19] = np.mean(close_4h[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_4h)):
            ema_4h[i] = close_4h[i] * alpha + ema_4h[i-1] * (1 - alpha)
    
    # Align daily CCI and 4h EMA to 1h
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(cci_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or not session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily CCI < -100 (oversold), price above 4h EMA (uptrend filter), volume spike
            if (cci_aligned[i] < -100 and close[i] > ema_4h_aligned[i] and vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: daily CCI > +100 (overbought), price below 4h EMA (downtrend filter), volume spike
            elif (cci_aligned[i] > 100 and close[i] < ema_4h_aligned[i] and vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: CCI returns to neutral (> -50) or trend turns down
            if (cci_aligned[i] > -50 or close[i] < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: CCI returns to neutral (< +50) or trend turns up
            if (cci_aligned[i] < 50 or close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_CCI_MeanReversion_WithVolume"
timeframe = "1h"
leverage = 1.0