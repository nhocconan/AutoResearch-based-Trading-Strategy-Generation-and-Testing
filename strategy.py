#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 2.0x 20-day avg volume
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 2.0x 20-day avg volume
# Exit when price crosses opposite 10-day Donchian level (mean reversion)
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining edge.
# Target: 15-25 trades/year on 1d (60-100 total over 4 years).
# Donchian channels provide robust breakout signals; 1w EMA50 filters counter-trend moves in bear markets.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in both bull (trend continuation) and bear (mean reversion after spikes) regimes.

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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channels on 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period Donchian channels for exit
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: >2.0x 20-day average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or np.isnan(donchian_high_10[i]) or 
            np.isnan(donchian_low_10[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1w_aligned[i]
        
        # Donchian levels
        upper_20 = donchian_high_20[i]
        lower_20 = donchian_low_20[i]
        upper_10 = donchian_high_10[i]
        lower_10 = donchian_low_10[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 10-day Donchian low (mean reversion)
            if curr_close < lower_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-day Donchian high (mean reversion)
            if curr_close > upper_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above 20-day high AND price > 1w EMA50 AND volume confirmation
            if curr_close > upper_20 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-day low AND price < 1w EMA50 AND volume confirmation
            elif curr_close < lower_20 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals