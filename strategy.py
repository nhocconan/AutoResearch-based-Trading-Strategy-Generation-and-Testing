#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA20) and volume confirmation (>2.0x 20-bar volume MA)
# Uses 1w HTF for stronger trend alignment, reducing whipsaws in ranging markets.
# Donchian breakouts capture strong momentum after consolidation.
# Volume confirmation ensures institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) with strong BTC/ETH performance.

name = "6h_Donchian20_Breakout_1wEMA20_Trend_VolumeConfirm_v1"
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
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(20) on 1w close
    ema_1w_20 = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_1w_20_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_20)
    
    # Calculate Donchian channels from previous 20 periods (6h)
    # Highest high of last 20 bars (excluding current)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lowest low of last 20 bars (excluding current)
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_20_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, above weekly EMA, and volume confirmation
            if curr_high > highest_high[i] and curr_close > ema_1w_20_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, below weekly EMA, and volume confirmation
            elif curr_low < lowest_low[i] and curr_close < ema_1w_20_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Donchian lower band or below weekly EMA
            if curr_low < lowest_low[i] or curr_close < ema_1w_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Donchian upper band or above weekly EMA
            if curr_high > highest_high[i] or curr_close > ema_1w_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals