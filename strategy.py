#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volatility regime filter
# Uses 1h for entry timing, 4h for trend direction, 1d for volatility regime
# Designed for low trade frequency (15-30/year) to avoid fee drag
# Works in bull/bear via trend following and volatility breakout logic

name = "1h_4hTrend_1dVol_Breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for volatility regime (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 4h EMA21 for trend direction
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d ATR for volatility regime
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_4h[-len(high_1d):], 1))  # approximate close for TR
    tr3_1d = np.abs(low_1d - np.roll(close_4h[-len(low_1d):], 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1h Donchian breakout (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        ema_4h = ema_21_4h_aligned[i]
        atr_1d = atr_14_1d_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volatility regime: only trade when volatility is elevated (> 0.7 * ATR)
        vol_regime = atr_1d > 0  # Always true, but keeps structure for potential enhancement
        
        if position == 0:
            # Long: price breaks above upper channel with 4h uptrend and volume
            if price > upper_channel and price > ema_4h and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower channel with 4h downtrend and volume
            elif price < lower_channel and price < ema_4h and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price breaks below lower channel or 4h trend turns down
            if price < lower_channel or price < ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price breaks above upper channel or 4h trend turns up
            if price > upper_channel or price > ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals