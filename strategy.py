#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses 1w HTF for pivot-based trend (bull/bear) and 1d for volume spike confirmation.
# Long when price breaks above 6h Donchian upper AND weekly pivot > previous weekly pivot (bullish bias) AND 1d volume > 2.0x 20-bar average.
# Short when price breaks below 6h Donchian lower AND weekly pivot < previous weekly pivot (bearish bias) AND 1d volume > 2.0x 20-bar average.
# Exit when price crosses 6h Donchian midline (mean of upper/lower).
# Discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Works in bull/bear via weekly pivot trend filter and volume confirmation to avoid false breakouts.

name = "6h_Donchian20_1wPivot_Trend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot point (typical price) for trend: (H+L+C)/3
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_pivot = pd.Series(typical_price_1w).rolling(window=2, min_periods=2).mean().values  # current vs previous
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly pivot trend: 1 if current pivot > previous pivot (bullish), -1 if < (bearish), 0 otherwise
    weekly_pivot_prev = np.roll(weekly_pivot_aligned, 1)
    weekly_pivot_prev[0] = np.nan
    weekly_trend = np.where(weekly_pivot_aligned > weekly_pivot_prev, 1,
                           np.where(weekly_pivot_aligned < weekly_pivot_prev, -1, 0))
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 6h Donchian channels (20-period)
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, donchian_window, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_trend[i]) or 
            np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_weekly_trend = weekly_trend[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian upper, bullish weekly trend, volume confirmation
            if (curr_high > donchian_upper[i] and 
                curr_weekly_trend == 1 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Donchian lower, bearish weekly trend, volume confirmation
            elif (curr_low < donchian_lower[i] and 
                  curr_weekly_trend == -1 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit condition: price crosses Donchian midline
            if curr_close < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses Donchian midline
            if curr_close > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals