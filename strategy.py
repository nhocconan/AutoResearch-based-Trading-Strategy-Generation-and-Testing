#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Primary: 1d timeframe for lower frequency and reduced fee drag (target 30-100 trades over 4 years)
# - HTF: 1w for trend direction (EMA50) to avoid counter-trend trades
# - Long: Price breaks above Donchian(20) high + weekly EMA50 uptrend + volume > 1.5x 20-day MA
# - Short: Price breaks below Donchian(20) low + weekly EMA50 downtrend + volume > 1.5x 20-day MA
# - Exit: Price reverts to Donchian(20) midpoint (mean reversion) or breaks opposite Donchian(10) level
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Weekly EMA50 filter avoids counter-trend trades in bear markets, Donchian breakouts capture momentum in trending markets

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
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
    
    # Calculate 1d Donchian channels (20-period for breakout, 10-period for exit)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    donchian_mid_20 = (high_20 + low_20) / 2.0
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly EMA50 direction
        weekly_uptrend = ema50_1w_aligned[i] > ema50_1w_aligned[max(0, i-5)]  # Rising over past week
        weekly_downtrend = ema50_1w_aligned[i] < ema50_1w_aligned[max(0, i-5)]  # Falling over past week
        
        # Volume confirmation: current volume > 1.5x 20-day MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high + weekly uptrend + volume spike
            if (close_1d[i] > high_20[i] and weekly_uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian(20) low + weekly downtrend + volume spike
            elif (close_1d[i] < low_20[i] and weekly_downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian(20) midpoint (mean reversion)
            # 2. Price breaks opposite Donchian(10) level (take profit/stop)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < donchian_mid_20[i] or  # Reverted to midpoint
                    close_1d[i] < low_10[i]              # Break below Donchian(10) low
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > donchian_mid_20[i] or  # Reverted to midpoint
                    close_1d[i] > high_10[i]             # Break above Donchian(10) high
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals