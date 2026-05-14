#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation (>1.8x 20-bar MA)
# Uses 12h HTF for stronger trend alignment than 6h, reducing whipsaws in ranging markets.
# Donchian breakouts capture strong momentum moves. Volume confirmation ensures institutional participation.
# Discrete sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "6h_Donchian20_Breakout_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(34) on 12h close
    ema_12h_34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_12h_34_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_34)
    
    # Donchian channel from 6h data (using previous 20 bars)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 34)  # Need 20 for Donchian, 34 for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_34_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, above 12h EMA, and volume confirmation
            if curr_high > donchian_high[i] and curr_close > ema_12h_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below 12h EMA, and volume confirmation
            elif curr_low < donchian_low[i] and curr_close < ema_12h_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Donchian low or below 12h EMA
            if curr_low < donchian_low[i] or curr_close < ema_12h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Donchian high or above 12h EMA
            if curr_high > donchian_high[i] or curr_close > ema_12h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals