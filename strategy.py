#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation
# Long when price breaks above upper band AND close > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below lower band AND close < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price crosses opposite Donchian band (mean reversion)
# Uses discrete position sizing (0.25) to limit fee drag and manage drawdown.
# Target: 15-25 trades/year on 1d (60-100 total over 4 years).
# Donchian channels provide structural breakout levels; 1w EMA50 filters counter-trend moves in bear markets.
# Volume confirmation ensures breakout validity, reducing false signals.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels from prior 20 days (using daily data)
    # We need to use daily OHLC to calculate proper Donchian bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Use prior 20 days (exclude current day) for Donchian calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min of prior 20 days (shifted by 1 to avoid look-ahead)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    # Rolling window of 20 prior days (not including current)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian bands to 1d timeframe (already aligned since we used 1d data)
    # But we need to shift by 1 to ensure we're using prior day's calculation
    # The shift(1) above already handles this
    
    # Volume confirmation: >1.5x 20-day average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1w_aligned[i]
        
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below lower band (mean reversion)
            if curr_close < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (mean reversion)
            if curr_close > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND close > 1w EMA50 AND volume confirmation
            if curr_close > upper_band and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower band AND close < 1w EMA50 AND volume confirmation
            elif curr_close < lower_band and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals