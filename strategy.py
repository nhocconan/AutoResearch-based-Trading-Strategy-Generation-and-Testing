#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation (>1.8x 20-bar MA)
# Donchian channels provide clear breakout levels. Breakout above upper band or below lower band with
# 1d EMA trend alignment and volume confirmation captures strong momentum moves. Works in bull markets
# via breakouts above upper band with uptrend, and in bear markets via breakdowns below lower band with downtrend.
# Target: 100-150 total trades over 4 years (25-38/year) with discrete sizing (0.25).

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Daily EMA(50) on 1d close
    daily_ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 4h timeframe
    daily_ema_50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema_50)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for EMA
    
    for i in range(start_idx, n):
        if np.isnan(daily_ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, above daily EMA, and volume confirmation
            if curr_close > highest_high[i] and curr_close > daily_ema_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, below daily EMA, and volume confirmation
            elif curr_close < lowest_low[i] and curr_close < daily_ema_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Donchian lower band or below daily EMA
            if curr_close < lowest_low[i] or curr_close < daily_ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Donchian upper band or above daily EMA
            if curr_close > highest_high[i] or curr_close > daily_ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals