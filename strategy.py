#!/usr/bin/env python3
"""
4h_PriceChannel_1dTrend_VolumeSpike - Hypothesis: Price channels (Donchian high/low of 20 periods) breakouts with volume spike and daily EMA(50) trend filter capture momentum in both bull and bear markets. Uses tight entry conditions to limit trades to 20-30/year, reducing fee drag while maintaining edge.
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
    
    # Get daily data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility filter (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 2.5x 20-period average on 4h (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0  # track holding period
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        upper_channel = high_max[i]
        lower_channel = low_min[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above upper channel with volume spike, price above daily EMA, and sufficient volatility
            if price > upper_channel and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume spike, price below daily EMA, and sufficient volatility
            elif price < lower_channel and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 3 bars (12 hours for 4h)
            if bars_since_entry < 3:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to lower channel or breaks below daily EMA
                if price <= lower_channel or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 3 bars (12 hours for 4h)
            if bars_since_entry < 3:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to upper channel or breaks above daily EMA
                if price >= upper_channel or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_PriceChannel_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0