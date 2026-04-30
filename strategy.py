#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (close > 1d EMA50) and volume confirmation.
# Long when price breaks above Donchian(20) upper band with 1d uptrend and volume > 1.8x 20-bar avg.
# Short when price breaks below Donchian(20) lower band with 1d downtrend and volume > 1.8x 20-bar avg.
# Exit on opposite Donchian band touch (mean reversion within the channel).
# Uses proven Donchian structure with strict volume confirmation (1.8x) and 1d EMA50 trend filter to limit trades.
# 1d EMA50 provides longer-term trend filter, reducing false signals in choppy markets and bear rallies.
# Timeframe: 6h, HTF: 1d as per experiment guidelines.

name = "6h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous 1d OHLC for completed 1d bar (no look-ahead)
    df_1d_prev = get_htf_data(prices, '1d')
    if len(df_1d_prev) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d_prev['high'].shift(1).values
    prev_low_1d = df_1d_prev['low'].shift(1).values
    prev_close_1d = df_1d_prev['close'].shift(1).values
    
    # Align 1d data to 6h timeframe (completed 1d bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_close_1d)
    
    # Donchian(20) channels from previous completed 1d bar (no look-ahead)
    # Upper = max(high, 20), Lower = min(low, 20)
    donchian_upper = pd.Series(prev_high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prev_low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d_prev, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d_prev, donchian_lower)
    
    # Volume confirmation: volume > 1.8x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_donchian_upper = donchian_upper_aligned[i]
        curr_donchian_lower = donchian_lower_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, uptrend (close > 1d EMA50), volume spike
            if (curr_close > curr_donchian_upper and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, downtrend (close < 1d EMA50), volume spike
            elif (curr_close < curr_donchian_lower and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Donchian lower (mean reversion)
            if curr_close <= curr_donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Donchian upper (mean reversion)
            if curr_close >= curr_donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals