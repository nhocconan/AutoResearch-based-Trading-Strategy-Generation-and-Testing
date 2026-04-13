#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
    # Long when 1d close > 1d EMA50 (uptrend) AND Bull Power > 0 AND 6h volume > 1.5x 20-period MA.
    # Short when 1d close < 1d EMA50 (downtrend) AND Bear Power < 0 AND 6h volume > 1.5x 20-period MA.
    # Exit when Bull/Bear Power crosses zero (momentum shift).
    # Uses Elder Ray (Bull/Bear Power) for momentum, daily EMA for trend, volume for confirmation.
    # Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - optional but good practice
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray Index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready or outside session (optional)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_spike = volume_6h_aligned[i] > 1.5 * vol_ma_6h_aligned[i]
        
        # Trend filter: daily close relative to daily EMA50
        daily_uptrend = close_1d[-1] > ema_50_1d[-1] if len(close_1d) > 0 else False  # Simplified: use aligned array
        daily_downtrend = close_1d[-1] < ema_50_1d[-1] if len(close_1d) > 0 else False
        
        # Better approach: use the aligned daily data at the 6h bar's corresponding time
        # Since we're using 6h bars, we need to check the trend at the corresponding daily bar
        # For simplicity in 6h timeframe, we'll use the most recent daily data available
        # But to avoid look-ahead, we'll use the aligned arrays properly
        
        # Actually, we should check if the daily close (from the most recent completed daily bar) 
        # is above/below the daily EMA50
        # Since we have aligned arrays, we can use the close_1d and ema_50_1d values
        # but we need to map the 6h bar to the correct daily bar
        
        # Simpler and correct approach: use the aligned daily close and EMA values
        # We need to get the daily close array aligned to 6h timeframe
        df_1d_for_close = get_htf_data(prices, '1d')
        close_1d_aligned = align_htf_to_ltf(prices, df_1d_for_close, df_1d_for_close['close'].values)
        ema_50_1d_for_close = pd.Series(df_1d_for_close['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1d_aligned_for_close = align_htf_to_ltf(prices, df_1d_for_close, ema_50_1d_for_close)
        
        daily_uptrend = close_1d_aligned[i] > ema_50_1d_aligned_for_close[i]
        daily_downtrend = close_1d_aligned[i] < ema_50_1d_aligned_for_close[i]
        
        # Elder Ray conditions
        bullish_momentum = bull_power_aligned[i] > 0
        bearish_momentum = bear_power_aligned[i] < 0
        exit_signal = (bull_power_aligned[i] * bear_power_aligned[i]) < 0  # Zero cross
        
        # Entry conditions
        if daily_uptrend and bullish_momentum and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif daily_downtrend and bearish_momentum and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif exit_signal and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0