#!/usr/bin/env python3
"""
4h_Donchian20_WeeklyTrend_VolumeSpike_v1
Hypothesis: 4h Donchian(20) breakouts aligned with weekly trend (price > weekly EMA20) and volume spikes capture strong momentum with low false breakouts. Weekly EMA filter ensures we only trade in the direction of higher timeframe trend, reducing whipsaws in sideways markets. Target: 25-35 trades/year on 4h timeframe.
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility filter (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 1.5x 50-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (1.5 * atr_ma)
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        weekly_trend = ema_20_1w_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper channel with volume spike, price above weekly EMA, and sufficient volatility
            if price > upper_channel and vol_spike and price > weekly_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume spike, price below weekly EMA, and sufficient volatility
            elif price < lower_channel and vol_spike and price < weekly_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to lower channel or breaks below weekly EMA
            if price < lower_channel or price < weekly_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to upper channel or breaks above weekly EMA
            if price > upper_channel or price > weekly_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_WeeklyTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0