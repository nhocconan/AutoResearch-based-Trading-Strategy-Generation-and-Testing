#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 200-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-day EMA for trend filter (long-term trend)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 4-hour data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 20-period Donchian channel on 4h data
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Calculate 4-hour ATR for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_ma20 = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    atr_ma20_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_ma20)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need EMA200, Donchian, volume MA20, ATR MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(atr_ma20_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema200_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR > 20-period ATR average (avoid low volatility)
        volatility_filter = atr_aligned[i] > atr_ma20_aligned[i]
        # Long-term trend filter: price above/below 200-day EMA
        trend_up = close[i] > ema200_4h[i]
        trend_down = close[i] < ema200_4h[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, volatility AND long-term uptrend
            if (close[i] > donchian_high_aligned[i] and volume_filter and volatility_filter and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, volatility AND long-term downtrend
            elif (close[i] < donchian_low_aligned[i] and volume_filter and volatility_filter and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or volatility drops
            if close[i] < donchian_low_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or volatility drops
            if close[i] > donchian_high_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_200DEMA_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0