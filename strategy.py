#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with weekly trend filter and volume confirmation
# In weekly bull (price > weekly EMA20): buy breakout above BB upper band
# In weekly bear (price < weekly EMA20): sell breakdown below BB lower band
# Volume confirmation: require volume > 1.5x 20-period average
# Bollinger Bands (20, 2) provide dynamic support/resistance
# Weekly EMA20 filter adapts to higher timeframe trend to avoid counter-trend trades
# Designed to work in both bull and bear markets by following the weekly trend
# Target: 20-50 total trades over 4 years (5-12/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly timeframe for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily data for Bollinger Bands and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Bollinger Bands
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_bull = close[i] > ema20_1w_aligned[i]
        is_bear = close[i] < ema20_1w_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        
        if position == 0:
            # Enter long in bull trend: price breaks above BB upper band with volume
            if is_bull and has_volume and price > bb_up:
                signals[i] = 0.25
                position = 1
            # Enter short in bear trend: price breaks below BB lower band with volume
            elif is_bear and has_volume and price < bb_low:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to BB middle (SMA) or opposite band
            if price < sma20[i]:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to BB middle (SMA) or opposite band
            if price > sma20[i]:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BollingerBand_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0