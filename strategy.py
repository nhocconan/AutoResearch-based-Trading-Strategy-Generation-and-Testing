#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper (20-period) AND price > 1d EMA50 AND volume > 1.5x 20-bar average.
# Short when price breaks below 4h Donchian lower (20-period) AND price < 1d EMA50 AND volume > 1.5x 20-bar average.
# Exit when price crosses 4h EMA20 (dynamic stop) OR opposite Donchian breakout occurs.
# Uses Donchian channels for structure (proven edge on SOL/ETH), 1d EMA for trend alignment, volume for conviction.
# Target: 25-35 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing strong trends.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) by trading with aligned 1d trend.

name = "4h_Donchian20_1dEMA50_VolumeBreakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Donchian and EMA calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 4h EMA20 for dynamic exit
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h primary timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 (50) + Donchian (20) + EMA20 (20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_donch_high = donchian_high_aligned[i]
        curr_donch_low = donchian_low_aligned[i]
        curr_ema_20 = ema_20_4h_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian upper + price > 1d EMA50 + volume confirmation
            if (curr_high > curr_donch_high and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower + price < 1d EMA50 + volume confirmation
            elif (curr_low < curr_donch_low and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below 20-period EMA OR opposite Donchian breakout
            if (curr_close < curr_ema_20) or \
               (curr_low < curr_donch_low):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above 20-period EMA OR opposite Donchian breakout
            if (curr_close > curr_ema_20) or \
               (curr_high > curr_donch_high):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals