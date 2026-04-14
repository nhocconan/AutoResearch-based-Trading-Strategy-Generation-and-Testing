#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian Channels with volume confirmation and trend filter.
# Long when price breaks above weekly upper Donchian Channel (20), volume > 1.5x average, and price above weekly EMA50 (trend filter).
# Short when price breaks below weekly lower Donchian Channel (20), volume > 1.5x average, and price below weekly EMA50.
# Exit when price returns to weekly Donchian middle (average of upper and lower) or volume drops below average.
# Uses weekly structure for major trend, daily for execution. Designed to capture strong trends while avoiding chop.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Donchian Channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for Donchian(20) and EMA(50)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channels (20)
    # Upper: highest high over past 20 weeks
    # Lower: lowest low over past 20 weeks
    # Middle: average of upper and lower
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    donch_middle = (donch_upper + donch_lower) / 2
    
    # Calculate EMA(50) for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to daily timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, donch_lower)
    donch_middle_aligned = align_htf_to_ltf(prices, df_1w, donch_middle)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need EMA50 and Donchian periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or
            np.isnan(donch_middle_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price relative to EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume and trend confirmation
            # Long: price breaks above upper Donchian AND volume confirmed AND price above EMA50
            if (close[i] > donch_upper_aligned[i] and 
                volume_confirmed and 
                price_above_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian AND volume confirmed AND price below EMA50
            elif (close[i] < donch_lower_aligned[i] and 
                  volume_confirmed and 
                  price_below_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle Donchian or volume drops below average
            if (close[i] <= donch_middle_aligned[i] or 
                volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle Donchian or volume drops below average
            if (close[i] >= donch_middle_aligned[i] or 
                volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Channels_Volume_EMAFilter_v1"
timeframe = "1d"
leverage = 1.0