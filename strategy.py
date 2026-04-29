#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 2.0x 24-bar avg
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 2.0x 24-bar avg
# Exit when price retouches Donchian midpoint or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-40 trades/year on 4h.
# Donchian channels provide robust price structure with proven edge in trending markets.
# 1d EMA50 filter ensures we only trade with the long-term trend, improving win rate.
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >2.0x 24-bar average volume (6 periods on 4h = 1 day)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Volume MA needs 24 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 1d EMA50 AND volume confirmation
            if curr_high > upper and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND price < 1d EMA50 AND volume confirmation
            elif curr_low < lower and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches midpoint or breaks below lower
            if curr_close <= mid or curr_low < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches midpoint or breaks above upper
            if curr_close >= mid or curr_high > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals