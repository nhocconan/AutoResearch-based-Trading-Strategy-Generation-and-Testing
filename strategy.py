#!/usr/bin/env python3
"""
4h Donchian Channel Breakout with Volume Spike and Daily Trend Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirmation
avoids false breakouts, while daily EMA50 filter ensures alignment with higher timeframe trend.
Designed for 25-50 trades/year on 4h timeframe. Works in bull markets via long breakouts
and bear markets via short breakouts when daily trend is down.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_50)
    
    # Donchian Channel (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (4h ATR)
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
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above EMA50 (uptrend)
            if price > upper and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below EMA50 (downtrend)
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

name = "4h_Donchian_Breakout_VolumeSpike_EMA50"
timeframe = "4h"
leverage = 1.0