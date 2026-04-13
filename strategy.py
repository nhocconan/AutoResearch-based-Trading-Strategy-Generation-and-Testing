#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator trend filter with 1w volume spike and Donchian(20) breakout.
    # Long when price breaks above Donchian(20) high AND Alligator is bullish (jaw < teeth < lips) AND 1w volume > 2x 20-period MA.
    # Short when price breaks below Donchian(20) low AND Alligator is bearish (jaw > teeth > lips) AND 1w volume > 2x 20-period MA.
    # Exit when price crosses the Alligator teeth (midline).
    # Uses discrete position sizing (0.25) to target 50-150 trades over 4 years.
    # Works in bull/bear via Alligator trend filter avoiding false breakouts in choppy markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w volume 20-period MA
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w volume MA to 12h timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    # Calculate Williams Alligator on 12h timeframe (SMAs of median price)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 2x 20-period average
        volume_spike = volume_1w_aligned[i] > 2.0 * vol_ma_1w_aligned[i]
        
        # Alligator trend conditions
        alligator_bullish = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        alligator_bearish = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high[i-1]  # Break above previous period high
        breakout_short = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # Exit conditions: price crosses Alligator teeth (midline)
        exit_long = close[i] < teeth[i]
        exit_short = close[i] > teeth[i]
        
        # Entry conditions
        if breakout_long and volume_spike and alligator_bullish and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and volume_spike and alligator_bearish and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "12h_1w_alligator_volume_donchian_v1"
timeframe = "12h"
leverage = 1.0