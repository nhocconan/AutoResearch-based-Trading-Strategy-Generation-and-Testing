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
    
    # Get daily data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr1.iloc[0] = 0
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian, volume MA, and daily EMA200
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr14_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        trend_up = close[i] > ema200_1d_aligned[i]
        trend_down = close[i] < ema200_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        atr_val = atr14_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + uptrend
            if close[i] > donchian_high[i] and vol_spike_val and trend_up:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + volume spike + downtrend
            elif close[i] < donchian_low[i] and vol_spike_val and trend_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian low or trend turns down
            if close[i] < donchian_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Donchian high or trend turns up
            if close[i] > donchian_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_v2"
timeframe = "4h"
leverage = 1.0