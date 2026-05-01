#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d volume spike confirmation
# Uses 4h EMA50 for trend direction (price > EMA50 = long bias, price < EMA50 = short bias)
# 1h Donchian breakout provides entry timing in direction of 4h trend
# 1d volume spike (>1.5x 20 EMA) confirms institutional participation
# Designed for low frequency (60-150 trades over 4 years) with clear structure
# Works in bull/bear: trend filter avoids counter-trend whipsaws, volume confirms breakout strength

name = "1h_Donchian20_4hEMA50_1dVolume_v1"
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
    
    # 4h HTF data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d HTF data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Donchian channels (20-period)
    # Upper channel = highest high of last 20 periods
    # Lower channel = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d volume spike filter: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ema_20_1d = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ema_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)  # Need EMA50 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout above upper channel + uptrend + volume spike
            if uptrend and close[i] > donchian_upper[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Donchian breakout below lower channel + downtrend + volume spike
            elif downtrend and close[i] < donchian_lower[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to midpoint of Donchian channel or opposite breakout
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
            exit_long = False
            if close[i] <= donchian_mid:  # Return to midpoint
                exit_long = True
            elif close[i] < donchian_lower[i] and volume_spike_1d_aligned[i]:  # Reverse breakout
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns to midpoint of Donchian channel or opposite breakout
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
            exit_short = False
            if close[i] >= donchian_mid:  # Return to midpoint
                exit_short = True
            elif close[i] > donchian_upper[i] and volume_spike_1d_aligned[i]:  # Reverse breakout
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals