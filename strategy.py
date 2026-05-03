#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime filter + 12h Donchian breakout with volume confirmation.
# Uses BB Width percentile to detect low volatility squeeze (breakout precursor) and 12h Donchian for direction.
# Long: BB Width < 20th percentile (squeeze) AND price breaks above 12h Donchian(20) upper band AND volume > 1.5x 20-period MA
# Short: BB Width < 20th percentile (squeeze) AND price breaks below 12h Donchian(20) lower band AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or BB Width > 50th percentile (squeeze end)
# Discrete sizing 0.25. Target: 50-120 total trades over 4 years (12-30/year).
# BB Width regime prevents whipsaw in choppy markets, Donchian provides clear breakout signals,
# volume confirmation reduces false breakouts. Works in bull via upside breakouts and in bear via downside breakouts.

name = "6h_BBWidth_Squeeze_12hDonchian_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian breakout
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Bollinger Band Width regime on 6h (20,2)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2 * std_20)
    lower_bb = ma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
    
    # BB Width percentile rank (lookback 50 periods)
    bb_width_rank = pd.Series(bb_width).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(bb_width_rank[i]) or np.isnan(ma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        upper_donch = donchian_upper_aligned[i]
        lower_donch = donchian_lower_aligned[i]
        bb_width_percentile = bb_width_rank[i]
        vol_spike = volume_spike[i]
        
        # Squeeze condition: BB Width < 20th percentile (low volatility)
        is_squeeze = bb_width_percentile < 0.20
        
        # Breakout conditions
        breakout_up = close_val > upper_donch
        breakout_down = close_val < lower_donch
        
        # Exit condition: BB Width > 50th percentile (squeeze ending) or opposite breakout
        squeeze_end = bb_width_percentile > 0.50
        
        # Entry logic
        if position == 0:
            # Long: squeeze AND upside breakout AND volume spike
            if is_squeeze and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: squeeze AND downside breakout AND volume spike
            elif is_squeeze and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: squeeze ending OR downside breakout
            if squeeze_end or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: squeeze ending OR upside breakout
            if squeeze_end or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals