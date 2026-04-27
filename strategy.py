#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Uses weekly trend to filter breakouts, reducing false signals in choppy markets.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Need at least 20 days of data for Donchian calculation
        if i < 20:
            signals[i] = 0.0
            continue
            
        # Calculate Donchian channels for previous day (using completed daily bar)
        idx_1d = i // 24  # Convert 1h index to approximate day index (24 hours per day)
        if idx_1d >= 20 and idx_1d < len(df_1d):
            # Get highest high and lowest low over past 20 days (excluding current day)
            start_idx_1d = max(0, idx_1d - 20)
            end_idx_1d = idx_1d  # Exclude current day
            if end_idx_1d > start_idx_1d:
                highest_high = np.max(df_1d['high'].iloc[start_idx_1d:end_idx_1d].values)
                lowest_low = np.min(df_1d['low'].iloc[start_idx_1d:end_idx_1d].values)
                
                # Long: price breaks above 20-day high with uptrend and volume
                if (close[i] > highest_high and 
                    close[i] > ema50_1w_aligned[i] and 
                    volume_filter[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 20-day low with downtrend and volume
                elif (close[i] < lowest_low and 
                      close[i] < ema50_1w_aligned[i] and 
                      volume_filter[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    # Hold current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0