#!/usr/bin/env python3
"""
1d Donchian Breakout with Volume Spike and Weekly EMA Trend Filter
Hypothesis: Weekly EMA defines long-term trend; breakouts of daily Donchian channel
with volume confirmation capture momentum in both bull and bear markets.
Designed for 7-25 trades/year on 1d timeframe.
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
    
    # Get weekly data for EMA20 (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA20 for trend filter
    ema_20 = pd.Series(df_w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_w, ema_20)
    
    # Daily Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above weekly EMA20 (uptrend)
            if price > upper and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below weekly EMA20 (downtrend)
            elif price < lower and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to lower Donchian or ATR trailing stop
            if price <= lower or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to upper Donchian or ATR trailing stop
            if price >= upper or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_VolumeSpike_WeeklyEMA20"
timeframe = "1d"
leverage = 1.0