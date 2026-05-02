#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakout captures sustained momentum moves, works in both bull and bear markets
# 1w EMA50 ensures we only trade in the direction of the primary trend, reducing whipsaws
# Volume spike (>1.5 x 20-period EMA) confirms breakout validity
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 30-100 total trades over 4 years (7-25/year) for optimal risk-adjusted returns
# Uses 1d primary timeframe with 1w HTF for trend alignment as specified in experiment #116150

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # 1w EMA50 calculation (trend filter)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for Donchian channel calculation
    if len(prices) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    # Upper channel: highest high over last 20 periods
    # Lower channel: lowest low over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50 (price above/below EMA)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper channel with volume confirmation and uptrend
            if high[i] > highest_high[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower channel with volume confirmation and downtrend
            elif low[i] < lowest_low[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian lower channel (reversal to downside)
            if low[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian upper channel (reversal to upside)
            if high[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals