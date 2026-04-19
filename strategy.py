#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1d_Trend_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - close_1d_prev))
    tr_1d = np.maximum(tr_1d, np.absolute(low_1d - close_1d_prev))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200 = ema_200_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high + above daily EMA200 + volume
            if price > upper and price > ema200 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below daily EMA200 + volume
            elif price < lower and price < ema200 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below Donchian low or ATR stop
            if price < lower or price < close[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above Donchian high or ATR stop
            if price > upper or price > close[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals