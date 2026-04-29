#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend + volume spike
# Williams Alligator identifies trend direction via SMAs (jaw/teeth/lips).
# 1d EMA34 filters for higher timeframe trend alignment.
# Volume spike confirms breakout strength.
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag.
# Works in bull/bear via trend filter and volatility-based entry.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (13,8,5 periods smoothed)
    # Jaw: 13-period SMA smoothed by 8 periods
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(sma_13).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA smoothed by 5 periods
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(sma_8).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA smoothed by 3 periods
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(sma_5).rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13, 8, 5, 20)  # warmup for EMA34, Alligator, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit: Alligator lines cross (lips < teeth) OR trend changes (price < 1d EMA34)
            if (curr_lips < curr_teeth) or (curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (lips > teeth) OR trend changes (price > 1d EMA34)
            if (curr_lips > curr_teeth) or (curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Alligator alignment: lips > teeth > jaw (bullish) OR lips < teeth < jaw (bearish)
            bullish_align = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)
            bearish_align = (curr_lips < curr_teeth) and (curr_teeth < curr_jaw)
            
            # Long entry: bullish alignment + price above 1d EMA34 + volume confirm
            if bullish_align and (curr_close > curr_ema_34_1d) and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: bearish alignment + price below 1d EMA34 + volume confirm
            elif bearish_align and (curr_close < curr_ema_34_1d) and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals