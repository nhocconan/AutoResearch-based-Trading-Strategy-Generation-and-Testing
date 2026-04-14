#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Channel Breakout with Weekly EMA Trend Filter and Volume Confirmation
# Uses weekly EMA (50) for trend direction to avoid counter-trend trades
# Breakout occurs when price breaks above/below 20-period Donchian channel
# Volume confirmation requires current volume > 1.5x 20-period average volume
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drift
# Works in both bull/bear markets by following the higher timeframe trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (50) for trend direction
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 20-period Donchian channel
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of weekly EMA
        above_weekly_ema = price > ema_weekly_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and uptrend filter
            if price > donchian_high[i] and vol > 1.5 * avg_volume[i] and above_weekly_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume confirmation and downtrend filter
            elif price < donchian_low[i] and vol > 1.5 * avg_volume[i] and not above_weekly_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches Donchian low (mean reversion) or trend changes
            if price <= donchian_low[i] or price < ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches Donchian high (mean reversion) or trend changes
            if price >= donchian_high[i] or price > ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_WeeklyEMA_Volume"
timeframe = "12h"
leverage = 1.0