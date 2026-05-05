#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above 1d Donchian upper(20) AND price > 1w EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Donchian lower(20) AND price < 1w EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 1d Donchian midpoint OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 30-100 total trades over 4 years (7-25/year)
# Donchian channels from 1d provide robust support/resistance; 1w EMA50 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1d_Donchian20_1wEMA50_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Donchian levels (though prices is already 1d, we need rolling window)
    # For Donchian on 1d timeframe, we can calculate directly from prices
    # But we still need to get 1w data for EMA50
    
    # Calculate 1d Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper, above 1w EMA50, volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower, below 1w EMA50, volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals