#!/usr/bin/env python3
"""
4h_PriceChannel_KeltnerBreakout_With_TrendAndVolume
Hypothesis: 4h price breaks above/below Keltner upper/lower band with volume spike and trend confirmation.
In bull markets, captures breakouts above upper band; in bear markets, captures breakdowns below lower band.
Trend filter uses 1d EMA34 to avoid counter-trend trades. Volume spike ensures momentum confirmation.
Designed for 20-40 trades/year to minimize fee drag while capturing strong directional moves.
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
    
    # Keltner Channel on 4h: 20-period ATR multiplier 2.0
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    keltner_upper = ma + (2.0 * atr)
    keltner_lower = ma - (2.0 * atr)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Trend filter: 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        keltner_up = keltner_upper[i]
        keltner_low = keltner_lower[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and uptrend
            if price > keltner_up and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike and downtrend
            elif price < keltner_low and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below middle band OR trend turns down
            if price < ma[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above middle band OR trend turns up
            if price > ma[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_PriceChannel_KeltnerBreakout_With_TrendAndVolume"
timeframe = "4h"
leverage = 1.0