#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above upper Donchian(20) AND close > 1w EMA34 AND volume > 1.5x 20-bar avg
# Short when price breaks below lower Donchian(20) AND close < 1w EMA34 AND volume > 1.5x 20-bar avg
# Exit when price crosses 1w EMA34 (trend change)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to avoid overtrading.
# Donchian channels provide robust structure for breakouts in both bull and bear markets.
# Weekly EMA34 ensures alignment with longer-term trend, reducing counter-trend trades.
# Volume confirmation filters low-participation false breakouts.

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) from 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Donchian and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 1w EMA34 (trend change)
            if curr_close < curr_ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1w EMA34 (trend change)
            if curr_close > curr_ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND close > 1w EMA34 AND volume confirmation
            if curr_close > curr_donch_high and curr_close > curr_ema34_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian AND close < 1w EMA34 AND volume confirmation
            elif curr_close < curr_donch_low and curr_close < curr_ema34_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals