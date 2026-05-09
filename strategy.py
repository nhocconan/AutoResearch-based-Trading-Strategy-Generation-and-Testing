#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeTrend_4hEMA200"
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
    
    # Get 1d data for trend filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for long-term trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d ATR for volatility regime (filter out low volatility periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr2[0] = tr1[0]  # first value
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 4h EMA200 for trend confirmation
    ema200_4h = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 4h volume spike filter
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 30)  # Need enough data for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema200_4h[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_1d_val = ema200_1d_aligned[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        ema200_4h_val = ema200_4h[i]
        vol_spike = volume_spike[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_ma_1d_val = atr_ma_1d_aligned[i]
        
        # Volatility regime filter: only trade when volatility is elevated
        vol_regime = atr_1d_val > atr_ma_1d_val * 0.8  # Avoid extremely low vol
        
        if position == 0:
            # Enter long: Price breaks above Donchian high + above 4h EMA200 + 1d uptrend + volume spike + vol regime
            if (close[i] > dh and 
                close[i] > ema200_4h_val and 
                close[i] > ema200_1d_val and 
                vol_spike and 
                vol_regime):
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low + below 4h EMA200 + 1d downtrend + volume spike + vol regime
            elif (close[i] < dl and 
                  close[i] < ema200_4h_val and 
                  close[i] < ema200_1d_val and 
                  vol_spike and 
                  vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low or 4h EMA200 turns down
            if close[i] < dl or close[i] < ema200_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high or 4h EMA200 turns up
            if close[i] > dh or close[i] > ema200_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals