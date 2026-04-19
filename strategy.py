#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_TrendVolume_Squeeze_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volatility (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily 200 EMA for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200 = ema200_1d_aligned[i]
        atr = atr14_1d_aligned[i]
        upper = high_max_20[i]
        lower = low_min_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility (squeeze)
        # Use 50-period ATR average to normalize
        if i >= 50:
            atr_ma_50 = pd.Series(atr14_1d_aligned[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            volatility_normal = atr > 0.5 * atr_ma_50 and atr < 2.0 * atr_ma_50
        else:
            volatility_normal = True
        
        if position == 0:
            # Long: break above upper Donchian with uptrend, volume, and normal volatility
            if price > upper and price > ema200 and volume_confirmed and volatility_normal:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with downtrend, volume, and normal volatility
            elif price < lower and price < ema200 and volume_confirmed and volatility_normal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below lower Donchian or trend reversal
            if price < lower or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above upper Donchian or trend reversal
            if price > upper or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals