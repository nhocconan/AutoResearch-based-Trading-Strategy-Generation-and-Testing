#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 12h EMA Trend Filter
Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and 12h EMA trend filter capture momentum moves while avoiding false breakouts. Designed for 20-50 trades/year on 4h timeframe, works in bull/bear markets by filtering with higher timeframe trend.
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
    
    # Get 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA34 on 12h close
    ema_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels on 4h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        ema = ema_12h_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above 12h EMA (uptrend)
            if price > upper and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below 12h EMA (downtrend)
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

name = "4h_Donchian_Breakout_VolumeSpike_12hEMA34"
timeframe = "4h"
leverage = 1.0