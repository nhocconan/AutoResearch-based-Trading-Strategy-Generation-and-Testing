#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit when price retests 10-day MA (mean reversion) or opposite Donchian level
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 7-25 trades/year on 1d timeframe.
# Donchian channels provide structural breakout levels, 1w EMA50 filters the major trend,
# volume confirmation ensures breakout validity. Works in bull via breakout continuation,
# in bear via breakdown continuation. Novelty: using 1d timeframe (lower frequency) with
# weekly trend filter to avoid overtrading and capture major moves.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels (using previous bar's data to avoid look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 10-period MA for exit (mean reversion target)
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ma10 = ma_10[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests 10-day MA or breaks below Donchian low
            if curr_low <= curr_ma10 or curr_close <= curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests 10-day MA or breaks above Donchian high
            if curr_high >= curr_ma10 or curr_close >= curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND price > 1w EMA50 AND volume confirmation
            if curr_high > curr_donchian_high and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND price < 1w EMA50 AND volume confirmation
            elif curr_low < curr_donchian_low and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals