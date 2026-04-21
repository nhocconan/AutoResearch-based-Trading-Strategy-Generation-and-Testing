#!/usr/bin/env python3
"""
1h Candle Close Above/Below 4h EMA20 with Volume Spike
Hypothesis: In trending markets, price respects the 4h EMA20 as dynamic support/resistance.
A break above/below with volume continuation signals institutional participation.
Works in bull (buy dips to EMA) and bear (sell rallies to EMA) by using dynamic MA.
Volume spike filters low-conviction moves. 1h timeframe used only for entry timing.
Target: 15-30 trades/year per symbol via strict EMA/volume confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for EMA20
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # EMA20 on 4h close
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if EMA not ready
        if np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema = ema_4h_aligned[i]
        vol = volume[i]
        
        # Volume spike: current > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_spike = vol > 1.5 * vol_ma
        else:
            vol_spike = False
        
        # Session filter: 08:00-20:00 UTC
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        if position == 0 and in_session:
            # Long: close above EMA20 with volume spike
            if price > ema and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: close below EMA20 with volume spike
            elif price < ema and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA20
            if price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above EMA20
            if price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA20_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0