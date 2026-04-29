#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA50 trend filter
# Uses Donchian channel from 1d timeframe for breakout signals
# Volume confirmation (>2.0x 20-period average) filters false breakouts
# 1w EMA50 trend filter ensures alignment with higher timeframe momentum
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
# Proven pattern: Donchian + volume + trend = SOLUSDT test Sharpe 1.10-1.38 (from research)
# Works in both bull and bear markets: trend filter avoids counter-trend trades, volume confirmation reduces false signals

name = "1d_Donchian20_VolumeConfirmation_1wEMA50_Trend"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) from 1d timeframe
    # We need to calculate this on 1d data first, then align
    # For simplicity, we'll calculate Donchian on 1d closes and align
    # But we need 1d OHLC data for proper Donchian
    
    # Since we're on 1d timeframe, we can calculate directly
    # But we need to use the prices DataFrame which is already 1d
    high_1d = high
    low_1d = low
    close_1d = close
    
    # Calculate Donchian channels (20-period high/low)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 20)  # Donchian, 1w EMA, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_20[i]
        curr_lower = lower_20[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band
            if curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band
            if curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian with volume confirmation and uptrend
            if vol_confirm and curr_high > curr_upper and curr_close > curr_ema_1w:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_lower and curr_close < curr_ema_1w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals