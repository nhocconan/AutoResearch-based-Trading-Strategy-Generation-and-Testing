#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price retraces to Donchian midpoint (mean of upper/lower) or opposite band
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide structural breakout levels, 1d EMA50 filters counter-trend moves,
# volume confirmation ensures breakout validity. Works in bull via breakout continuation,
# in bear via breakdown continuation. Novelty: combining Donchian with 1d EMA50 and volume spike.

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retraces to midpoint or breaks below lower band
            if curr_low <= curr_mid or curr_close <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price retraces to midpoint or breaks above upper band
            if curr_high >= curr_mid or curr_close >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND price > 1d EMA50 AND volume confirmation
            if curr_high > curr_upper and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below lower band AND price < 1d EMA50 AND volume confirmation
            elif curr_low < curr_lower and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals