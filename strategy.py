#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (>1.5x average)
# Donchian channels provide clear trend-following structure. Breakout above/below 20-day high/low
# with 1w EMA50 trend filter ensures we trade with the higher timeframe trend. Volume confirmation
# avoids low-conviction breakouts. Works in bull/bear via trend filter. Target: 30-100 total trades
# over 4 years (7-25/year) with discrete sizing 0.25-0.30.

name = "1d_Donchian20_1wEMA50_Trend_Volume_v1"
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
    
    # Calculate 20-period Donchian channels (using daily high/low)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_donchian_high = high_rolling_max[i]
        curr_donchian_low = low_rolling_min[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Donchian breakout
            if curr_volume_spike:
                # Bullish: price breaks above 20-day high + price above 1w EMA50
                if curr_high > curr_donchian_high and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below 20-day low + price below 1w EMA50
                elif curr_low < curr_donchian_low and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below 20-day low (trend reversal)
            if curr_low < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high (trend reversal)
            if curr_high > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals