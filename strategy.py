#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: Uses 4h Donchian channel (20-period) breakouts filtered by 12h EMA50 trend and volume spike (>1.5x average). Enters long when price breaks above upper Donchian band AND 12h close > 12h EMA50 (uptrend) AND volume > 1.5x average. Enters short when price breaks below lower Donchian band AND 12h close < 12h EMA50 (downtrend) AND volume > 1.5x average. Exits when price reverts to 12h EMA50 (mean reversion to trend) OR opposite Donchian breakout occurs. Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year to avoid fee drag while capturing strong trending moves. Works in both bull and bear markets via 12h trend filter and volume confirmation to avoid low-conviction breakouts.
"""

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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    # Upper band = max(high, 20), Lower band = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20), 12h EMA50 (50), volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_12h_val = ema_50_12h_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: price breakout above upper band (long) or below lower band (short) with trend and volume
            # Long: price > upper band AND 12h uptrend AND volume confirmation
            long_condition = (close_val > upper_band) and (close_val > ema_12h_val) and vol_conf
            # Short: price < lower band AND 12h downtrend AND volume confirmation
            short_condition = (close_val < lower_band) and (close_val < ema_12h_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to 12h EMA50 (mean reversion to trend) OR opposite breakout
            exit_condition = (close_val <= ema_12h_val) or (close_val < lower_band)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to 12h EMA50 (mean reversion to trend) OR opposite breakout
            exit_condition = (close_val >= ema_12h_val) or (close_val > upper_band)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0