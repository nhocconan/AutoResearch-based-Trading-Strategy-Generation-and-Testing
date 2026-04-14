#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channels (20-period) for breakout signals.
# Weekly Donchian high/low provide strong institutional support/resistance levels.
# Entry only when price breaks above/below weekly Donchian channel with volume confirmation (>1.5x 20-day average).
# Trend filter: price must be above/below 50-day EMA to avoid counter-trend trades.
# Exit when price returns to weekly midline (average of Donchian high/low) or opposite Donchian band is touched.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag in both bull and bear markets.
# Works in trending markets via breakouts and in ranging markets via mean reversion to midline.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian High: 20-period rolling maximum
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian Low: 20-period rolling minimum
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Donchian Midline: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly Donchian levels to daily timeframe (wait for weekly close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Load daily data ONCE for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above weekly Donchian high or below weekly Donchian low
            # Only trade in direction of 50-day EMA (trend filter)
            
            # Long: price breaks above weekly Donchian high AND price above 50-day EMA (bullish)
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly Donchian low AND price below 50-day EMA (bearish)
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly midline or touches weekly Donchian low
            if (close[i] <= donchian_mid_aligned[i] or 
                close[i] <= donchian_low_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly midline or touches weekly Donchian high
            if (close[i] >= donchian_mid_aligned[i] or 
                close[i] >= donchian_high_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_20wDonchian_50EMA_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0