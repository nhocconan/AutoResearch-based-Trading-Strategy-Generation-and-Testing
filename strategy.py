#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high AND price > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below 20-day Donchian low AND price < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit when price retouches the opposite Donchian level (long exit at Donchian low, short exit at Donchian high)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 15-25 trades/year on 1d.
# Donchian channels provide proven structure for breakout trading with edge in both trending and ranging markets.
# 1w EMA34 filter ensures we only trade with the higher timeframe trend, improving win rate and reducing whipsaws.
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.
# This strategy focuses on BTC and ETH as primary targets, avoiding SOL-only bias.

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
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period Donchian channels using completed daily bars only
    # We need to use rolling window on daily data, but align to 1d timeframe properly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 and Donchian to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_34 = ema_34_1w_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND price > 1w EMA34 AND volume confirmation
            if curr_high > donch_high and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND price < 1w EMA34 AND volume confirmation
            elif curr_low < donch_low and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches Donchian low
            if curr_low <= donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches Donchian high
            if curr_high >= donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals