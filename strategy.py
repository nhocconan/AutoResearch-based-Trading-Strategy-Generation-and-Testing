#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts for trend capture with weekly EMA filter for regime alignment
# Volume > 1.5x average confirms institutional participation
# Discrete position sizing (0.25) with opposite Donchian(10) exit for risk control
# Designed for ~15-25 trades/year to minimize fee drag while capturing strong trends
# Works in bull/bear via trend filter and controlled exit logic

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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period for entry, 10-period for exit)
    # Donchian upper = max(high, lookback_period)
    # Donchian lower = min(low, lookback_period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_20_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_20_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_10_upper = high_series.rolling(window=10, min_periods=10).max().values
    donchian_10_lower = low_series.rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_20_upper[i]) or np.isnan(donchian_20_lower[i]) or 
            np.isnan(donchian_10_upper[i]) or np.isnan(donchian_10_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donch_20_up = donchian_20_upper[i]
        curr_donch_20_low = donchian_20_lower[i]
        curr_donch_10_up = donchian_10_upper[i]
        curr_donch_10_low = donchian_10_lower[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 10-period Donchian lower (tight stop for risk control)
            if curr_close < curr_donch_10_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 10-period Donchian upper (tight stop for risk control)
            if curr_close > curr_donch_10_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above 20-period Donchian upper, 1w EMA50 up-trend, volume confirmed
            if curr_high > curr_donch_20_up and curr_close > curr_ema50_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-period Donchian lower, 1w EMA50 down-trend, volume confirmed
            elif curr_low < curr_donch_20_low and curr_close < curr_ema50_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals