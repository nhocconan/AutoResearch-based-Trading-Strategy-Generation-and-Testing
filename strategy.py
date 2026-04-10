#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Primary: 1d timeframe for lower frequency and reduced fee drag
# - HTF: 1w for trend direction (price above/below 50-week EMA)
# - Long: Price breaks above 20-day Donchian upper + price > 50-week EMA + volume > 1.5x 20-day MA
# - Short: Price breaks below 20-day Donchian lower + price < 50-week EMA + volume > 1.5x 20-day MA
# - Exit: Price reverts to 20-day Donchian midpoint (mean reversion)
# - Position sizing: 0.25 (discrete level)
# - Target: 30-100 total trades over 4 years (7-25/year) - within 1d sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, EMA filter avoids counter-trend trades

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian Channel (20-period)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate 1w 50-period EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 50-week EMA
        uptrend = close_1d[i] > ema_50_1w_aligned[i]
        downtrend = close_1d[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + uptrend + volume spike
            if (close_1d[i] > highest_20[i] and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + downtrend + volume spike
            elif (close_1d[i] < lowest_20[i] and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price reverts to Donchian midpoint (mean reversion)
            if position == 1:  # Long position
                if close_1d[i] < midpoint_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_1d[i] > midpoint_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals