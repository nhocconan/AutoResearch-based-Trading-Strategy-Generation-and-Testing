#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly trend filter with daily Donchian breakout and volume confirmation
# Uses 1-week Donchian(10) to determine primary trend direction
# Enters on daily Donchian(20) breakout in trend direction with volume > 1.5x 20-day average
# Exits when price crosses back below/above daily Donchian(10) opposite band
# Designed to capture major trend moves while avoiding counter-trend whipsaws
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years)

name = "1d_WeeklyTrend_DonchianBreakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian(10) for trend determination
    # Upper band: highest high of last 10 weekly periods
    # Lower band: lowest low of last 10 weekly periods
    dh_1w = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    dl_1w = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    
    # Align weekly Donchian bands to daily timeframe
    dh_1w_aligned = align_htf_to_ltf(prices, df_1w, dh_1w)
    dl_1w_aligned = align_htf_to_ltf(prices, df_1w, dl_1w)
    
    # Daily Donchian channels for entry/exit
    dh_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dh_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    dl_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need daily Donchian(20) and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(dh_1w_aligned[i]) or np.isnan(dl_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_upper = dh_1w_aligned[i]
        weekly_lower = dl_1w_aligned[i]
        daily_high_20 = dh_20[i]
        daily_low_20 = dl_20[i]
        daily_high_10 = dh_10[i]
        daily_low_10 = dl_10[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Determine trend based on weekly Donchian
        # Uptrend: price above weekly upper band
        # Downtrend: price below weekly lower band
        # Sideways: between bands (no trades)
        if price > weekly_upper:
            trend = 1  # uptrend
        elif price < weekly_lower:
            trend = -1  # downtrend
        else:
            trend = 0  # sideways/no trend
        
        if position == 0:
            # Enter long: daily breaks above 20-day high in uptrend with volume
            if trend == 1 and price > daily_high_20 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: daily breaks below 20-day low in downtrend with volume
            elif trend == -1 and price < daily_low_20 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below 10-day low
            if price < daily_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above 10-day high
            if price > daily_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals