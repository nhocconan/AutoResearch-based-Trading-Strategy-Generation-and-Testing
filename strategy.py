#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Trend
Hypothesis: Donchian channel breakouts with volume confirmation and 1d EMA34 trend filter on 12h timeframe.
Long when price breaks above 20-period high with volume spike in uptrend (close > 1d EMA34).
Short when price breaks below 20-period low with volume spike in downtrend (close < 1d EMA34).
Uses 12h timeframe to reduce trade frequency and focus on strong trends in both bull and bear markets.
Target: 15-25 trades/year to minimize fee drag while capturing significant moves.
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
    
    # Donchian channel: 20-period high and low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper band with volume spike in uptrend
            if (price > upper and
                vol_spike and
                price > ema34):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike in downtrend
            elif (price < lower and
                  vol_spike and
                  price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below lower band or trend reverses
            if price < lower or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above upper band or trend reverses
            if price > upper or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0