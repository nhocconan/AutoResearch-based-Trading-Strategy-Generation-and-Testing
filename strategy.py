#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above Donchian(20) high + 1d volume > 1.5x average + price > 1w EMA(50)
# - Short when price breaks below Donchian(20) low + 1d volume > 1.5x average + price < 1w EMA(50)
# - Exit when price returns to Donchian midpoint or trend reverses
# - Designed for trend following with volume confirmation to avoid false breakouts
# - Target: 25-40 trades/year to stay within fee limits

name = "4h_Donchian20_1dVolume_1wTrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_factor = vol_ma_1d_aligned[i] / 6.0 if vol_ma_1d_aligned[i] > 0 else 0
        volume_filter = volume[i] > 1.5 * volume_factor
        
        if position == 0:
            # Look for long entry: uptrend + breakout above Donchian high + volume
            if close[i] > ema_50_1w_aligned[i] and close[i] > donchian_high[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend + breakout below Donchian low + volume
            elif close[i] < ema_50_1w_aligned[i] and close[i] < donchian_low[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to midpoint or trend reverses
            if close[i] < donchian_mid[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to midpoint or trend reverses
            if close[i] > donchian_mid[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals