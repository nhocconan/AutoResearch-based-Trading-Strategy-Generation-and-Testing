#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and volatility filter
# Uses 1d ATR for volatility regime and 1w EMA for trend filter
# Designed to work in bull (breakouts) and bear (avoid false breakouts via volatility filter)
# Target: 20-40 trades/year to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly and daily data for trend and volatility filters
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA(12) for long-term trend filter
    close_1w = df_1w['close'].values
    ema_12_1w = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_12_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_12_1w)
    
    # Daily ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h price data for Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 4h volume filter (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_12_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend_1w = ema_12_1w_aligned[i]
        atr = atr_14_1d_aligned[i]
        
        # Volatility filter: avoid extremely low or high volatility regimes
        atr_ma_50 = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_50) and (atr < 3.0 * atr_ma_50)
        
        if position == 0:
            # Enter long on Donchian breakout with volume and trend filter
            if (price > highest_high[i]) and vol_ratio[i] > 1.8 and vol_filter and (price > ema_trend_1w):
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown with volume and trend filter
            elif (price < lowest_low[i]) and vol_ratio[i] > 1.8 and vol_filter and (price < ema_trend_1w):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakdown or volatility spike
            if (price < lowest_low[i]) or (atr > 4.0 * atr_ma_50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout or volatility spike
            if (price > highest_high[i]) or (atr > 4.0 * atr_ma_50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeTrendFilter_VolRegime_v1"
timeframe = "4h"
leverage = 1.0