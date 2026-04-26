#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_1dVolumeConfirm
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly trend (price > weekly EMA50) and confirmed by daily volume spike (>2.0x 20-bar avg) captures institutional momentum with controlled frequency. Weekly trend filter ensures directional bias in bull/bear markets, daily volume confirms participation, and discrete sizing (0.25) minimizes fee churn. Targets 12-30 trades/year (50-120 over 4 years) by requiring confluence of HTF trend, breakout, and volume.
"""

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
    
    # Get weekly data for HTF trend (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian(20) on 6h close (using prior 20 bars to avoid look-ahead)
    # Upper band = max(high of prior 20 bars), Lower band = min(low of prior 20 bars)
    high_shift = np.concatenate([[np.nan] * 20, high[:-20]]) if len(high) >= 20 else np.full_like(high, np.nan)
    low_shift = np.concatenate([[np.nan] * 20, low[:-20]]) if len(low) >= 20 else np.full_like(low, np.nan)
    
    # Rolling max/min of shifted arrays
    donchian_upper = pd.Series(hift_shift).rolling(window=20, min_periods=20).max().values if len(high) >= 20 else np.full_like(high, np.nan)
    donchian_lower = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values if len(low) >= 20 else np.full_like(low, np.nan)
    
    # Simplified: use rolling window on original with min_periods, then shift
    donchian_upper_raw = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower_raw = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = np.concatenate([[np.nan] * 20, donchian_upper_raw[:-20]]) if len(high) >= 20 else np.full_like(high, np.nan)
    donchian_lower = np.concatenate([[np.nan] * 20, donchian_lower_raw[:-20]]) if len(low) >= 20 else np.full_like(low, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) + weekly EMA (50) + daily vol MA (20)
    start_idx = max(50, 20, 20) + 20  # 20 for lookback, 50 for weekly EMA warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma_20_1d_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period daily average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with weekly trend and volume
            # Long: price breaks above upper band with uptrend (close > weekly EMA50) and volume spike
            long_signal = (high_val > upper_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below lower band with downtrend (close < weekly EMA50) and volume spike
            short_signal = (low_val < lower_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below lower band (contrarian exit) or weekly trend breaks
            if close_val < lower_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper band or weekly trend breaks
            if close_val > upper_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_1dVolumeConfirm"
timeframe = "6h"
leverage = 1.0