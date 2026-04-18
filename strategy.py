#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly EMA trend filter and volume confirmation.
# Long when price breaks above 1d Donchian upper (20) with volume > 1.5x 20-day average and price > weekly EMA(34).
# Short when price breaks below 1d Donchian lower (20) with volume > 1.5x 20-day average and price < weekly EMA(34).
# Exit when price returns to 1d Donchian middle (midpoint of upper/lower).
# Uses Donchian channels for structure, volume surge for conviction, weekly EMA for trend filter.
# Designed for ~10-20 trades/year per symbol.
name = "1d_Donchian_20_WeeklyEMA34_Volume_Filter"
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
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) on 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper (20-period high)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian middle (midpoint)
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2.0
    
    # Align Donchian levels to 1d timeframe (no shift needed as same timeframe)
    donch_high_20_aligned = donch_high_20
    donch_low_20_aligned = donch_low_20
    donch_mid_20_aligned = donch_mid_20
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # EMA(34) on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.5 * 20-day average (approximate with 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(donch_mid_20_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = donch_high_20_aligned[i]
        low_val = donch_low_20_aligned[i]
        mid_val = donch_mid_20_aligned[i]
        ema_val = ema_34_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume surge and above weekly EMA
            if close_val > high_val and vol_filter and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume surge and below weekly EMA
            elif close_val < low_val and vol_filter and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian middle
            if close_val <= mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian middle
            if close_val >= mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals