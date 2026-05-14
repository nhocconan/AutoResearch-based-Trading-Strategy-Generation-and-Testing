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
    
    # Get daily data for 14-period EMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA14 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema14_1d = close_1d_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    ema14_1d_aligned = align_htf_to_ltf(prices, df_1d, ema14_1d)
    
    # Get 4h data for Donchian channel (price channel)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(10) channel
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donch_high_10 = high_4h_series.rolling(window=10, min_periods=10).max().values
    donch_low_10 = low_4h_series.rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_10_aligned = align_htf_to_ltf(prices, df_4h, donch_high_10)
    donch_low_10_aligned = align_htf_to_ltf(prices, df_4h, donch_low_10)
    
    # Volume filter: current volume > 1.8x 10-period average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # ATR(10) for volatility filter and stop
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema14_1d_aligned[i]) or np.isnan(donch_high_10_aligned[i]) or 
            np.isnan(donch_low_10_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume and above daily EMA14
            if close[i] > donch_high_10_aligned[i] and volume_filter[i] and close[i] > ema14_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low with volume and below daily EMA14
            elif close[i] < donch_low_10_aligned[i] and volume_filter[i] and close[i] < ema14_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low OR ATR-based stop
            if close[i] < donch_low_10_aligned[i] or close[i] < (high[max(0, i-1)] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high OR ATR-based stop
            if close[i] > donch_high_10_aligned[i] or close[i] > (low[max(0, i-1)] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian10_4h_EMA14_Trend_Volume"
timeframe = "4h"
leverage = 1.0