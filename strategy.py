#SBATCH --job-name=1d_breakout_trend_volume
#SBATCH --output=slurm-%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=4GB

#!/usr/bin/env python3
"""
1d_breakout_trend_volume_v1
Hypothesis: On daily timeframe, combine Donchian channel breakout (20-period) 
with weekly trend filter (price above/below weekly SMA50) and volume confirmation.
Long when price breaks above Donchian upper band, weekly trend is up, and volume surges.
Short when price breaks below Donchian lower band, weekly trend is down, and volume surges.
Exit when price crosses opposite Donchian band or volume drops below average.
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
Works in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on daily
    donchian_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Weekly trend filter: price vs weekly SMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    # Trend: 1 if price > SMA50, -1 if price < SMA50
    weekly_trend = np.full(len(close_1w), 0)
    for i in range(len(close_1w)):
        if not np.isnan(close_1w[i]) and not np.isnan(sma50_1w[i]):
            if close_1w[i] > sma50_1w[i]:
                weekly_trend[i] = 1
            elif close_1w[i] < sma50_1w[i]:
                weekly_trend[i] = -1
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below lower Donchian band or volume drops below average
            if close[i] < lower_band[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above upper Donchian band or volume drops below average
            if close[i] > upper_band[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper Donchian band, weekly trend up, volume surge
            if (close[i] > upper_band[i] and 
                weekly_trend_aligned[i] == 1 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower Donchian band, weekly trend down, volume surge
            elif (close[i] < lower_band[i] and 
                  weekly_trend_aligned[i] == -1 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals