#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel from prior 1w period: long on break above upper band in uptrend, short on break below lower band in downtrend
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Designed for 1d timeframe to capture primary trend swings with controlled trade frequency (~10-20 trades/year)
# Works in both bull and bear markets by aligning with 1w trend (EMA50) to avoid counter-trend trades

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
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
    
    # Get 1w data for Donchian channel calculation (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channel from prior 1w bar (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (delayed by one 1w bar for look-ahead avoidance)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 20-period average volume for confirmation (on 1d data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: reverse signal on opposite Donchian level break or trend change
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend turns down (price < EMA50)
            if curr_low < curr_donchian_low or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend turns up (price > EMA50)
            if curr_high > curr_donchian_high or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian in uptrend (price > EMA50)
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > curr_donchian_high:  # Break above upper band
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below lower Donchian in downtrend (price < EMA50)
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < curr_donchian_low:  # Break below lower band
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals