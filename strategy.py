#!/usr/bin/env python3
name = "1d_3ATR_Donchian_Breakout_Trend_1w_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data for Donchian and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Donchian(20) on 1D
    donch_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) on 1D
    tr1 = df_1d['high'][1:].values - df_1d['low'][1:].values
    tr2 = np.abs(df_1d['high'][1:].values - df_1d['close'][:-1].values)
    tr3 = np.abs(df_1d['low'][1:].values - df_1d['close'][:-1].values)
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = np.zeros(len(df_1d))
    atr_14[:14] = np.nan
    for i in range(14, len(df_1d)):
        atr_14[i] = np.nanmean(tr_1d[i-13:i+1])
    
    # EMA(50) on 1W for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x 20-period average on 1D
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align indicators to 1D timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions
        breakout_long = close[i] > donch_high_20_aligned[i] + 3 * atr_14_aligned[i]
        breakout_short = close[i] < donch_low_20_aligned[i] - 3 * atr_14_aligned[i]
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        volume_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: bullish breakout in uptrend with volume
            if breakout_long and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout in downtrend with volume
            elif breakout_short and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend changes
            if close[i] < donch_low_20_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend changes
            if close[i] > donch_high_20_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals