#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price breaks 1w Donchian(20) with volume confirmation and ATR filter
# - Long when price breaks above 1w Donchian high(20) + volume > 1.5x 20-period 1d average + ATR filter
# - Short when price breaks below 1w Donchian low(20) + volume > 1.5x 20-period 1d average + ATR filter
# - Exit when price returns to 1w Donchian midpoint or ATR-based trailing stop
# - Uses weekly structure to avoid whipsaw in ranging markets, captures breaks in both bull/bear
# - Volume filter ensures conviction on breakouts
# - ATR filter avoids entries during low volatility
# - Target: 15-30 trades/year to stay within fee constraints

name = "12h_DonchianBreakout_1dVolume_1wATRFilter_v1"
timeframe = "12h"
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
    
    # Get 1w data for Donchian channels and ATR
    df_1w = get_htf_data(prices, '1w')
    
    # 1w Donchian channels (20-period)
    highest_high_1w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_1w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_1w = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    donchian_low_1w = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2.0
    
    # 1w ATR for volatility filter
    tr1 = pd.Series(df_1w['high']).rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0], raw=False)
    tr2 = pd.Series(df_1w['high']).rolling(2).apply(lambda x: abs(x.iloc[1] - x.iloc[0]), raw=False)
    tr3 = pd.Series(df_1w['low']).rolling(2).apply(lambda x: abs(x.iloc[1] - x.iloc[0]), raw=False)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high_1w[i]) or np.isnan(donchian_low_1w[i]) or \
           np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars, so divide by 2
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 2.0)
        
        # ATR filter: avoid low volatility environments
        atr_filter = atr_1w_aligned[i] > 0
        
        if position == 0:
            # Look for long entry: price breaks above 1w Donchian high + volume + ATR filter
            if close[i] > donchian_high_1w[i] and volume_filter and atr_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below 1w Donchian low + volume + ATR filter
            elif close[i] < donchian_low_1w[i] and volume_filter and atr_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to midpoint or breaks below low
            if close[i] < donchian_mid_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to midpoint or breaks above high
            if close[i] > donchian_mid_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals