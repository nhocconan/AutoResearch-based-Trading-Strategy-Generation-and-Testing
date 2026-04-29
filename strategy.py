#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses weekly EMA50 to filter only bullish breakouts in uptrend and bearish breakdowns in downtrend.
# Donchian channel breakouts capture sustained momentum moves.
# Volume confirmation (>1.5x 20-period average) ensures breakout conviction.
# Designed for ~10-25 trades/year on 1d timeframe to minimize fee drag while capturing major trends.
# Works in both bull and bear markets via 1w trend filter - only trades breakouts in trend direction.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v2"
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
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 1d data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band (mean reversion)
            if curr_close < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (mean reversion)
            if curr_close > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: bullish breakout above upper Donchian in uptrend (price > 1w EMA50)
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > curr_highest_20:  # Breakout above upper band
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: bearish breakdown below lower Donchian in downtrend (price < 1w EMA50)
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < curr_lowest_20:  # Breakdown below lower band
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals