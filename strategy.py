#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation
# 4h Donchian breakout provides directional bias from higher timeframe structure
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades in bear markets
# Volume spike (>1.8x 20-period average) confirms institutional participation and reduces false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete position sizing (0.20) balances return potential with fee minimization
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
# Works in both bull and bear: trend filter adapts to market regime, volume confirmation adds robustness

name = "1h_Donchian_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1d EMA50 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_high_4h = donchian_high_aligned[i]
        curr_low_4h = donchian_low_aligned[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        vol_spike = curr_volume > 1.8 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low OR breaks 1d EMA50 trend
            if curr_close < curr_low_4h or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high OR breaks 1d EMA50 trend
            if curr_close > curr_high_4h or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above 4h Donchian high AND above 1d EMA50 AND volume spike
            if curr_high > curr_high_4h and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low AND below 1d EMA50 AND volume spike
            elif curr_low < curr_low_4h and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals