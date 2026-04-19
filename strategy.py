#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_KeltnerBreakout_Volume_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for Keltner channel width
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA20 for Keltner middle line
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Keltner channels
    upper_keltner = ema20_1d_aligned + 2.0 * atr_1d_aligned
    lower_keltner = ema20_1d_aligned - 2.0 * atr_1d_aligned
    
    # 1d ADX for trend strength filter
    plus_dm = np.maximum(high_1d - np.roll(high_1d, 1), 0)
    minus_dm = np.maximum(np.roll(low_1d, 1) - low_1d, 0)
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), plus_dm, 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), minus_dm, 0)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d_adx = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d_adx
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d_adx
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 20)  # EMA20, ADX14, VolMA20
    
    for i in range(start_idx, n):
        if np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or \
           np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx = adx_1d_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: only trade when ADX > 20 (trending market)
        trend_ok = adx > 20
        
        if position == 0:
            # Long: price breaks above upper Keltner + volume + trend
            if price > upper_keltner[i] and volume_ok and trend_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + volume + trend
            elif price < lower_keltner[i] and volume_ok and trend_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle line (EMA20) or volatility drops
            if price < ema20_1d_aligned[i] or adx < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle line (EMA20) or volatility drops
            if price > ema20_1d_aligned[i] or adx < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals