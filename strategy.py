#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime filter + 12h Donchian(20) breakout + volume confirmation
# BB Width < 20th percentile = ranging market (fade at bands)
# BB Width > 80th percentile = trending market (breakout continuation)
# In ranging: long at lower band, short at upper band with volume confirmation
# In trending: breakout above upper Donchian(20) or below lower Donchian(20) with volume confirmation
# Uses 12h HTF for Donchian calculation to reduce noise, 6h for BB Width regime
# Discrete sizing 0.25 to manage drawdown, target 50-150 total trades over 4 years
# Works in bull via breakout continuation, works in bear via mean reversion in ranging markets

name = "6h_BBWidth_DonchianBreakout_12hVolume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donchian_high_20 = high_12h_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_12h_series.rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian channels to 6h timeframe (wait for completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # Calculate Bollinger Bands and Width on 6h close
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate BB Width percentiles (using 50-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values * 100
    
    # Volume confirmation: volume > 1.3 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(bb_width_pct[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime detection: BB Width percentile
            is_ranging = bb_width_pct[i] < 30  # Lower 30% = ranging market
            is_trending = bb_width_pct[i] > 70  # Upper 30% = trending market
            
            # In ranging market: mean reversion at Bollinger Bands
            if is_ranging:
                # Long at lower BB with volume spike
                if close[i] <= lower_bb[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at upper BB with volume spike
                elif close[i] >= upper_bb[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # In trending market: Donchian breakout continuation
            elif is_trending:
                # Long on breakout above Donchian high with volume spike
                if close[i] > donchian_high_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short on breakout below Donchian low with volume spike
                elif close[i] < donchian_low_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below SMA(20) or opposite BB touch
            if close[i] < sma_20[i] or close[i] >= upper_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above SMA(20) or opposite BB touch
            if close[i] > sma_20[i] or close[i] <= lower_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals