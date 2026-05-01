#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar MA)
# Donchian breakout captures strong momentum moves. 1d EMA50 ensures alignment with higher-timeframe trend
# to avoid counter-trend trades. Volume confirmation filters weak breakouts. Discrete sizing (0.30)
# balances profit potential with risk control. Target: 75-200 total trades over 4 years (19-50/year).

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
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, above 1d EMA50, volume confirmation
            if curr_high > donchian_high[i] and curr_close > ema_1d_50_aligned[i] and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low, below 1d EMA50, volume confirmation
            elif curr_low < donchian_low[i] and curr_close < ema_1d_50_aligned[i] and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below Donchian low or below 1d EMA50
            if curr_low < donchian_low[i] or curr_close < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on price above Donchian high or above 1d EMA50
            if curr_high > donchian_high[i] or curr_close > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals