#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation (1.5x 20-bar MA) + 1d EMA50 trend filter
# Long when price breaks above Donchian upper channel AND volume spike AND price > 1d EMA50
# Short when price breaks below Donchian lower channel AND volume spike AND price < 1d EMA50
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 trades over 4 years.
# Donchian channels provide clear structural breaks, volume confirms conviction, 1d EMA50 filters counter-trend trades.

name = "4h_Donchian20_Breakout_VolumeSpike_1dEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Donchian breakout conditions
        upper_break = curr_close > high_ma_20[i-1]  # Break above previous upper channel
        lower_break = curr_close < low_ma_20[i-1]   # Break below previous lower channel
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above upper channel, volume spike, price above 1d EMA50
            if upper_break and vol_spike and curr_close > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower channel, volume spike, price below 1d EMA50
            elif lower_break and vol_spike and curr_close < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below lower channel (reversal signal)
            if curr_close < low_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above upper channel (reversal signal)
            if curr_close > high_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals