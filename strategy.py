#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d Donchian Breakout
# Uses Bollinger Band Width percentile to detect low volatility squeezes (regime filter)
# Only takes Donchian(20) breakouts in the direction of 1d trend when BBW is in lowest 20th percentile
# Works in both bull and bear markets by trading breakouts from low volatility regimes
# Volume confirmation (>1.5x 20-period EMA) ensures institutional participation
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag

name = "6h_BBW_Regime_Donchian_1dTrend_Volume"
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
    
    # Get 1d data for Donchian trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Calculate Bollinger Bands and Band Width on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Calculate BB Width percentile rank (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(donchian_high_1d_aligned[i]) or 
            np.isnan(donchian_low_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: BB Width in lowest 20th percentile (low volatility squeeze)
        low_volatility_regime = bb_width_percentile[i] <= 0.20
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Donchian breakout signals with 1d trend filter
        if position == 0:
            # Long: price breaks above 1d Donchian high + low volatility regime + volume spike
            if (close[i] > donchian_high_1d_aligned[i] and 
                low_volatility_regime and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low + low volatility regime + volume spike
            elif (close[i] < donchian_low_1d_aligned[i] and 
                  low_volatility_regime and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 20-period 6h SMA OR volatility expands (BBW > 80th percentile)
            if (close[i] < sma_20[i] or bb_width_percentile[i] > 0.80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 20-period 6h SMA OR volatility expands (BBW > 80th percentile)
            if (close[i] > sma_20[i] or bb_width_percentile[i] > 0.80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals