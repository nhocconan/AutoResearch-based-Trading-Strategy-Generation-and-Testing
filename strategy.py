#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA50 trend filter + Donchian(20) breakout + volume confirmation
# Weekly EMA50 provides robust trend filter for daily timeframe, reducing whipsaw in both bull and bear markets.
# Donchian(20) breakout captures strong momentum moves with clear entry/exit levels.
# Volume confirmation (2.0x 20-period average) ensures institutional participation and reduces false breakouts.
# Discrete sizing 0.25 minimizes fee churn while maintaining adequate position sizing.
# Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe.

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    # Using prior 20 periods to avoid look-ahead
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # warmup for Donchian and volume calculations
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_high_ma_20 = high_ma_20[i]
        curr_low_ma_20 = low_ma_20[i]
        curr_vol_ma_20 = vol_ma_20[i]
        
        # Skip if volume data not available
        if np.isnan(curr_vol_ma_20) or curr_vol_ma_20 == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * curr_vol_ma_20)
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above Donchian upper band AND above 1w EMA50 (uptrend)
                if curr_close > curr_high_ma_20 and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower band AND below 1w EMA50 (downtrend)
                elif curr_close < curr_low_ma_20 and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian lower band or below 1w EMA50
            if curr_close < curr_low_ma_20 or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian upper band or above 1w EMA50
            if curr_close > curr_high_ma_20 or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals