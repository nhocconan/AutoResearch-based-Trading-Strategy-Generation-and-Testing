#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for ATR-based volatility regime (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Load daily data for Donchian channels and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly ATR(14) for volatility regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volatility regime: High volatility when weekly ATR > 1.5 * its 20-period MA
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_20)
    high_volatility = atr_1w_aligned > 1.5 * atr_ma_20_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high in high volatility regime AND above EMA50 (uptrend)
            if (close[i] > donchian_high_aligned[i] and 
                high_volatility[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low in high volatility regime AND below EMA50 (downtrend)
            elif (close[i] < donchian_low_aligned[i] and 
                  high_volatility[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite Donchian level
            if position == 1:
                # Exit long: Price closes below Donchian low
                if close[i] < donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian20_1wATR_VolRegime_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0