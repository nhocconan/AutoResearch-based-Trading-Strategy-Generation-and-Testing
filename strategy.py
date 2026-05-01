#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 20-bar Donchian high, price > 1d EMA34, volume > 1.5x 20-bar avg
# Short when price breaks below 20-bar Donchian low, price < 1d EMA34, volume > 1.5x 20-bar avg
# Uses discrete sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Donchian channels provide clear structure, EMA34 filters counter-trend noise, volume confirms conviction.

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Donchian(20) channels
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > high_ma_20[i-1]  # Close above previous period's high
        breakout_down = curr_close < low_ma_20[i-1]  # Close below previous period's low
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high, above 1d EMA34, volume confirmation
            if breakout_up and curr_close > ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, below 1d EMA34, volume confirmation
            elif breakout_down and curr_close < ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below Donchian low or below 1d EMA34
            if curr_close < low_ma_20[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above Donchian high or above 1d EMA34
            if curr_close > high_ma_20[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals