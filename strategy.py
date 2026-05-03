#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Donchian breakouts capture strong momentum moves. 1w EMA50 ensures we only trade in the
# direction of the weekly trend. Volume spike confirms institutional participation.
# This strategy works in both bull and bear markets by aligning with the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels (20-period) on 1d
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get current values
        donchian_high = highest_20[i]
        donchian_low = lowest_20[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(donchian_high) or np.isnan(donchian_low) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine trend regime: bull if close > 1w EMA50, bear if close < 1w EMA50
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Generate signals
        if position == 0:
            # Long: price breaks above Donchian high in bull trend with volume spike
            if is_bull_trend and close_val > donchian_high and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in bear trend with volume spike
            elif is_bear_trend and close_val < donchian_low and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or trend changes to bear
            if close_val < donchian_low or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or trend changes to bull
            if close_val > donchian_high or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals