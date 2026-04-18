#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_1dEMA34
Hypothesis: Price breaks above/below Donchian channels (20-period high/low) with volume spike and daily EMA34 trend filter. Works in both bull/bear markets by only taking breakouts in the direction of the daily trend, reducing false reversals. Target: 25-40 trades/year to minimize fee drag while capturing strong directional moves.
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
    
    # Daily EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and uptrend (price > daily EMA34)
            if (price > upper and vol_spike and price > ema34):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume spike and downtrend (price < daily EMA34)
            elif (price < lower and vol_spike and price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below daily EMA34 OR breaks below lower band (reversal)
            if price < ema34 or price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above daily EMA34 OR breaks above upper band (reversal)
            if price > ema34 or price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_1dEMA34"
timeframe = "4h"
leverage = 1.0