#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - 12h Donchian channel (20-period) as price structure
# - Breakout above upper band (bullish) or below lower band (bearish)
# - 1d EMA50 trend filter to avoid counter-trend trades
# - Volume spike (>2x 20-period average) for confirmation
# - Designed for low trade frequency (~20-40/year) to minimize fee drag on 12h
# - Works in bull/bear markets by aligning with 1d trend

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    donch_upper_aligned = align_htf_to_ltf(prices, df_12h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_12h, donch_lower)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + 1d uptrend + volume spike
            long_cond = (close[i] > donch_upper_aligned[i] and 
                        ema_50_1d_aligned[i] > close_1d[-1] if len(close_1d) > 0 else False and  # Simplified trend check
                        volume_spike[i])
            
            # Short: price breaks below Donchian lower + 1d downtrend + volume spike
            short_cond = (close[i] < donch_lower_aligned[i] and 
                         ema_50_1d_aligned[i] < close_1d[-1] if len(close_1d) > 0 else False and  # Simplified trend check
                         volume_spike[i])
            
            # Fix trend check: compare current EMA to previous EMA
            if i > 0:
                ema_prev = ema_50_1d_aligned[i-1]
                ema_curr = ema_50_1d_aligned[i]
                long_cond = (close[i] > donch_upper_aligned[i] and 
                            ema_curr > ema_prev and
                            volume_spike[i])
                short_cond = (close[i] < donch_lower_aligned[i] and 
                             ema_curr < ema_prev and
                             volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower (reversal signal)
            if close[i] < donch_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper (reversal signal)
            if close[i] > donch_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals