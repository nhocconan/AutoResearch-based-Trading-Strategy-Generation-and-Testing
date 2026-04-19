#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian20_TurnVolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    dc_upper = high_series.rolling(window=20, min_periods=20).max().values
    dc_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = dc_upper[i]
        lower = dc_lower[i]
        trend = ema50_12h_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above upper Donchian with volume and bullish 12h trend
            if price > upper and volume_confirmed and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and bearish 12h trend
            elif price < lower and volume_confirmed and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below lower Donchian or trend turns bearish
            if price < lower or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above upper Donchian or trend turns bullish
            if price > upper or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals