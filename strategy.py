#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v3
# Hypothesis: 1d strategy using weekly Donchian channels with volume confirmation.
# Long: Price breaks above weekly Donchian H20, volume > 1.3x 20-period average.
# Short: Price breaks below weekly Donchian L20, volume > 1.3x 20-period average.
# Exit: Price crosses weekly Donchian midpoint (mean reversion to center).
# Uses weekly structure for trend, volume to filter false breakouts, mean reversion exit.
# Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Donchian channels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian H20 and L20
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_h20 = rolling_max(high_1w, 20)
    donchian_l20 = rolling_min(low_1w, 20)
    donchian_mid = (donchian_h20 + donchian_l20) / 2.0
    
    # Align HTF Donchian levels to 1d timeframe (wait for completed weekly bar)
    donchian_h20_aligned = align_htf_to_ltf(prices, df_1w, donchian_h20)
    donchian_l20_aligned = align_htf_to_ltf(prices, df_1w, donchian_l20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_h20_aligned[i]) or np.isnan(donchian_l20_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly H20, volume confirmed
            if (high[i] > donchian_h20_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly L20, volume confirmed
            elif (low[i] < donchian_l20_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals