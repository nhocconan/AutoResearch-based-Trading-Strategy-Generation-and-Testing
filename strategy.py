#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d price is above weekly pivot (bullish bias) AND 6h volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian(20) low AND 1d price is below weekly pivot (bearish bias) AND 6h volume > 1.5x 20-period volume SMA
# - Exit: opposite Donchian breakout or volume drops below average
# - Uses 6h for Donchian and volume, 1d for weekly pivot calculation
# - Weekly pivot provides structural bias from higher timeframe
# - Volume confirmation ensures breakouts have conviction
# - Donchian breakouts capture sustained moves in both bull and bear markets
# - Target: 12-30 trades/year to minimize fee drag while capturing meaningful moves

name = "6h_1d_weekly_pivot_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for weekly pivot calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate weekly pivot points from 1d data (using prior week's high/low/close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close (using 5-day week approximation)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Pre-compute Donchian channels for 6h data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA for 6h data (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close (aligned) for pivot comparison
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_high[i-1]  # Break above prior period's high
        breakout_short = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Weekly pivot bias
        above_pivot = close_1d_aligned[i] > weekly_pivot_aligned[i]
        below_pivot = close_1d_aligned[i] < weekly_pivot_aligned[i]
        
        # Exit conditions: opposite breakout or volume drops below average
        exit_long = close[i] < donchian_low[i-1] or volume[i] < volume_sma_20[i]
        exit_short = close[i] > donchian_high[i-1] or volume[i] < volume_sma_20[i]
        
        # Trading logic
        if vol_confirm:
            # Long: Donchian breakout above weekly pivot
            if breakout_long and above_pivot:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout below weekly pivot
            elif breakout_short and below_pivot:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals