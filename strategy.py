#!/usr/bin/env python3
name = "12h_Vortex_Trend_Plus_Volume_1dFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, min_periods=100, adjust=False).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Load 1d data ONCE for volume filter (volume spike detection)
    vol_1d = df_1d['volume'].values
    vol_ma50_1d = pd.Series(vol_1d).rolling(window=50, min_periods=50).mean().values
    vol_ma50_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma50_1d)
    
    # Calculate Vortex Indicator on 12h data (requires high, low, close)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align length
    
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # Position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Vortex)
    start_idx = 150
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(vi_plus[i]) or 
            np.isnan(vi_minus[i]) or np.isnan(vol_ma50_1d_aligned[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_1d_ema = close[i] > ema100_1d_aligned[i]
        price_below_1d_ema = close[i] < ema100_1d_aligned[i]
        volume_spike = volume[i] > vol_ma50_1d_aligned[i] * 2.0
        vi_bullish = vi_plus[i] > vi_minus[i]
        vi_bearish = vi_plus[i] < vi_minus[i]
        
        if position == 0:
            # Long: VI bullish + price above 1d EMA + volume spike
            if vi_bullish and price_above_1d_ema and volume_spike:
                signals[i] = position_size
                position = 1
            # Short: VI bearish + price below 1d EMA + volume spike
            elif vi_bearish and price_below_1d_ema and volume_spike:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Vortex reversal OR price crosses 1d EMA
            if position == 1:
                if vi_bearish or price_below_1d_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if vi_bullish or price_above_1d_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals