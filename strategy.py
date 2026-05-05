#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above weekly Donchian high (20) AND 1d volume > 2.0 * avg_volume(20) AND 1d chop > 61.8 (range market)
# Short when price breaks below weekly Donchian low (20) AND 1d volume > 2.0 * avg_volume(20) AND 1d chop < 38.2 (trending market)
# Exit when price crosses weekly Donchian midpoint OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe
# Weekly Donchian provides robust structure from higher timeframe
# Volume spike confirms breakout strength and reduces false signals
# Chop regime filter adapts to market conditions: mean reversion in range, trend following in trending
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume)

name = "1d_Donchian20_Weekly_VolumeSpike_ChopRegime"
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
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_high = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1w_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly Donchian levels to 1d timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-period average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Calculate 1d choppiness index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)
    
    # Regime filters: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    chop_range = chop > 61.8
    chop_trending = chop < 38.2
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high, volume confirmation, in range market (CHOP > 61.8)
            if close[i] > donchian_high_aligned[i] and volume_confirm[i] and chop_range[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below weekly Donchian low, volume confirmation, in trending market (CHOP < 38.2)
            elif close[i] < donchian_low_aligned[i] and volume_confirm[i] and chop_trending[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals