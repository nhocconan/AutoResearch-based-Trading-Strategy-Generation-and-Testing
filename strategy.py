#!/usr/bin/env python3
"""
4h_Bollinger_Breakout_With_TrendAndVolume
Hypothesis: 4h price breaks above/below Bollinger Bands with volume spike and trend confirmation.
In bull markets, captures breakouts above upper band; in bear markets, captures breakdowns below lower band.
Trend filter uses 1d EMA50 to avoid counter-trend trades. Volume spike ensures momentum confirmation.
Designed for 15-30 trades/year to minimize fee drift while capturing strong directional moves.
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
    
    # Bollinger Band on 4h: 20-period SMA, 2.0 std
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma + (2.0 * std)
    lower = sma - (2.0 * std)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Trend filter: 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = upper[i]
        lower_band = lower[i]
        ema50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and uptrend
            if price > upper_band and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike and downtrend
            elif price < lower_band and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below middle band OR trend turns down
            if price < sma[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above middle band OR trend turns up
            if price > sma[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Bollinger_Breakout_With_TrendAndVolume"
timeframe = "4h"
leverage = 1.0