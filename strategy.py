#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Volume_Spike_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close']
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Load 1d data for Donchian channel breakout
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    
    # Donchian channel: 10-period high/low on daily
    donchian_high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_10)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_10)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        ema_12h_trend = ema_21_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and bullish 12h trend
            if close_val > donchian_high and vol_spike and (close_val > ema_12h_trend):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and bearish 12h trend
            elif close_val < donchian_low and vol_spike and (close_val < ema_12h_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below Donchian low or trend turns bearish
            if close_val < donchian_low or (close_val < ema_12h_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above Donchian high or trend turns bullish
            if close_val > donchian_high or (close_val > ema_12h_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals