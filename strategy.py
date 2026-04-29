#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channel from prior 12h period: long on break above upper band in uptrend, short on break below lower band in downtrend
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Designed for 12h timeframe to capture medium-term swings with controlled trade frequency (~12-30 trades/year)
# Works in both bull and bear markets by aligning with 1d trend (EMA34) to avoid counter-trend trades

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get prior 12h data for Donchian channel calculation (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) from prior 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band = max(high, 20), Lower band = min(low, 20)
    upper_band = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe (delayed by one 12h bar for look-ahead avoidance)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # Calculate 20-period average volume for confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # EMA34 warmup (longest lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_upper = upper_band_aligned[i]
        curr_lower = lower_band_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: reverse signal on opposite Donchian band break or trend change
        if position == 1:  # Long position
            # Exit: price breaks below lower band or trend turns down (close < EMA34)
            if curr_low < curr_lower or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper band or trend turns up (close > EMA34)
            if curr_high > curr_upper or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above upper band in uptrend (close > EMA34)
            if vol_confirm and curr_close > curr_ema34_1d:
                if curr_high > curr_upper:  # Break above upper band
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below lower band in downtrend (close < EMA34)
            elif vol_confirm and curr_close < curr_ema34_1d:
                if curr_low < curr_lower:  # Break below lower band
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals