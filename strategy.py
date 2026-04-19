#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w ADX25 trend filter and Donchian20 breakout with volume confirmation.
# Enters only during 08-20 UTC session. Uses strict conditions to limit trades (~10-20/year) and avoid overtrading.
# Trend-following in bull/bear markets, avoids false signals via ADX filter and volume spike requirement.
name = "1d_1w_ADX25_Donchian20_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for ADX25 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX25
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_sum = pd.Series(tr).rolling(window=25, min_periods=25).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=25, min_periods=25).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=25, min_periods=25).sum().values
    
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=25, min_periods=25).mean().values
    
    adx_25_1w = adx
    adx_25_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_25_1w)
    
    # Get 1d data for Donchian20 breakout (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Donchian channels: 20-period high/low
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(adx_25_1w_aligned[i]) or np.isnan(high_20_1d_aligned[i]) or 
            np.isnan(low_20_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above ADX25 threshold AND breaks 1d Donchian high with volume
            if (adx_25_1w_aligned[i] > 25 and 
                close[i] > high_20_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above ADX25 threshold AND breaks 1d Donchian low with volume
            elif (adx_25_1w_aligned[i] > 25 and 
                  close[i] < low_20_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d Donchian low
            if close[i] < low_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d Donchian high
            if close[i] > high_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals