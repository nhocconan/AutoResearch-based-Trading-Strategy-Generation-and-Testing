#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with weekly volume confirmation and trend filter
# Long: price breaks above upper BB + price > weekly EMA50 + volume > 2x weekly average
# Short: price breaks below lower BB + price < weekly EMA50 + volume > 2x weekly average
# Exit: price crosses back through middle BB (20 SMA)
# Weekly filters reduce noise and false breakouts, targeting 10-25 trades/year.
# Designed to work in both bull and bear markets by requiring strong volume and trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend and volume filters
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Weekly EMA50 for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Weekly volume average (20-period)
    vol_avg_weekly = pd.Series(volume_weekly).rolling(window=20, min_periods=20).mean().values
    vol_avg_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_avg_weekly)
    
    # Daily Bollinger Bands (20, 2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Middle BB = 20 SMA
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    stddev = pd.Series(close).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma20 + 2 * stddev
    lower_bb = sma20 - 2 * stddev
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg_weekly_aligned[i]) or \
           np.isnan(sma20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend
        is_uptrend = close[i] > ema50_weekly_aligned[i]
        is_downtrend = close[i] < ema50_weekly_aligned[i]
        
        # Weekly volume confirmation (volume > 2x weekly average)
        has_volume = volume[i] > (2 * vol_avg_weekly_aligned[i])
        
        price = close[i]
        
        if position == 0:
            # Long entry: price breaks above upper BB + uptrend + volume
            long_signal = (price > upper_bb[i]) and is_uptrend and has_volume
            
            # Short entry: price breaks below lower BB + downtrend + volume
            short_signal = (price < lower_bb[i]) and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle BB (20 SMA)
            if price < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle BB (20 SMA)
            if price > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BollingerBreakout_WeeklyTrendVolume"
timeframe = "1d"
leverage = 1.0