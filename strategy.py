#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeConfirmation
Hypothesis: On 1d timeframe, Donchian channel (20) breakouts with 1-week EMA50 trend filter and volume confirmation capture institutional breakouts in both bull and bear markets. Donchian channels identify key structural breakouts. In bull markets, we buy upside breaks with uptrend; in bear markets, we sell downside breaks with downtrend. Volume confirmation ensures breakout validity. Targets 7-25 trades/year to minimize fee drag while maintaining edge via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20) on 1d
    # We need to compute this on 1d data, but we can use rolling on the 1d-resampled equivalent
    # Since we have 1d timeframe prices, we can compute directly
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 50)  # Donchian, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_1w_val = ema_50_1w_aligned[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above upper Donchian with uptrend (close > EMA50_1w) and volume confirmation
            long_signal = (high_val > high_20_val) and (close_val > ema_50_1w_val) and volume_confirmed
            # Short: price breaks below lower Donchian with downtrend (close < EMA50_1w) and volume confirmation
            short_signal = (low_val < low_20_val) and (close_val < ema_50_1w_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below lower Donchian (exit long)
            if low_val < low_20_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA50_1w
            elif close_val < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above upper Donchian (exit short)
            if high_val > high_20_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA50_1w
            elif close_val > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0