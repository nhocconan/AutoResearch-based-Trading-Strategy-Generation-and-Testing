#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts for structural moves, filtered by weekly EMA50 trend
# Volume > 1.8x 20-period average confirms institutional participation
# Discrete position sizing (0.25) with Donchian mid-point exit to reduce whipsaw
# Target: < 25 trades/year to minimize fee drag while capturing strong moves in BTC/ETH

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    # Upper channel = highest high of last 20 periods
    # Lower channel = lowest low of last 20 periods
    # Middle channel = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume warmup
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_dc_middle = donchian_middle[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Skip if any required data is NaN
        if (np.isnan(curr_dc_upper) or np.isnan(curr_dc_lower) or np.isnan(curr_dc_middle) or 
            np.isnan(curr_ema50_1w) or np.isnan(curr_vol_ma)):
            signals[i] = 0.0
            continue
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Donchian middle (mean reversion to reduce whipsaw)
            if curr_close < curr_dc_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Donchian middle (mean reversion to reduce whipsaw)
            if curr_close > curr_dc_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above Donchian upper, 1w EMA50 up-trend, volume confirmed
            if curr_high > curr_dc_upper and curr_close > curr_ema50_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower, 1w EMA50 down-trend, volume confirmed
            elif curr_low < curr_dc_lower and curr_close < curr_ema50_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals