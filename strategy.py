#2025-06-05
# Hypothesis: 6h timeframe with weekly Donchian channel breakout and daily volume confirmation.
# In bull markets, price breaks above weekly high with volume; in bear markets, breaks below weekly low with volume.
# Weekly filter reduces whipsaw, volume confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year), size 0.25.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for weekly lookback
        return np.zeros(n)
    
    # Load weekly data once for Donchian channel (20-week lookback)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    # Use pandas rolling with min_periods to avoid look-ahead
    high_series = pd.Series(high_weekly)
    low_series = pd.Series(low_weekly)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Load daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    volume_daily = df_daily['volume'].values
    
    # Calculate daily volume average (20-period)
    volume_series = pd.Series(volume_daily)
    vol_ma_daily = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Align daily volume MA to 6h timeframe
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any data is not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_ma = vol_ma_daily_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + volume above average
            if price > upper_band and prices['volume'].iloc[i] > vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + volume above average
            elif price < lower_band and prices['volume'].iloc[i] > vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to the opposite Donchian band (mean reversion within channel)
            if position == 1 and price < lower_band:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian_Breakout_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0