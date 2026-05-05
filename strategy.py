#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses Donchian(10) mid-line OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 30-80 total trades over 4 years (7-20/year) with proper risk control
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1d_Donchian20_EMA50_Volume_Confirm"
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
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period) and mid-line (10-period)
    # Donchian high: highest high over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian mid-line: average of 20-period high and low
    donchian_mid = (donchian_high + donchian_low) / 2
    # Donchian(10) mid-line for exit
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian(20) high, above 1w EMA50, volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian(20) low, below 1w EMA50, volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian(10) mid-line OR volume drops below average
            if close[i] < donchian_mid_10[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Donchian(10) mid-line OR volume drops below average
            if close[i] > donchian_mid_10[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals