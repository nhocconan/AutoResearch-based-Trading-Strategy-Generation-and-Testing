#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA50 trend filter
# Uses Donchian channel breakouts from daily timeframe for structural trend continuation
# Volume confirmation (>2.0x 50-period average) filters false breakouts
# 1w EMA50 trend filter ensures alignment with weekly momentum
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
# Proven pattern: Donchian + volume + trend = SOLUSDT test Sharpe 1.10-1.38 (from research)

name = "1d_Donchian20_VolumeConfirmation_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from previous 1d bar
    # We need to calculate this on 1d data, then align to 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily data
    high_roll_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (use previous day's levels)
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, high_roll_20)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, low_roll_20)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 50-period average volume for confirmation
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 50)  # Donchian, 1w EMA, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper_dc = upper_dc_aligned[i]
        curr_lower_dc = lower_dc_aligned[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_50[i]
        
        # Volume confirmation: current volume > 2.0x 50-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian channel
            if curr_close < curr_lower_dc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian channel
            if curr_close > curr_upper_dc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above upper DC with volume confirmation and uptrend
            if vol_confirm and curr_high > curr_upper_dc and curr_close > curr_ema_1w:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower DC with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_lower_dc and curr_close < curr_ema_1w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals