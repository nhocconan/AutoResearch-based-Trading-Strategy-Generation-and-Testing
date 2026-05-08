#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA10 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel in uptrend (close > EMA10) with volume > 1.5x 20-day average.
# Short when price breaks below lower Donchian channel in downtrend (close < EMA10) with volume > 1.5x 20-day average.
# Exit when price crosses the 10-day EMA (trend reversal signal).
# Uses Donchian channels from daily timeframe, EMA10 from weekly for trend, volume confirmation.
# Designed to capture sustained trends in both bull and bear markets with proper risk control.
# Target: 10-25 trades/year to minimize fee decay while capturing major moves.

name = "1d_Donchian_20_1wEMA10_Volume"
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
    
    # Get weekly data for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 10-period EMA on weekly close
    ema_10 = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period high/low) on daily data
    # Using rolling window on price arrays directly
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly EMA10 to daily timeframe
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    
    # Pre-compute session filter (00-24 UTC - full day for 1d timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for daily, but keep for consistency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if outside trading session (though for 1d this is always true)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_10_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume
            # Long when price breaks above upper Donchian channel in uptrend (close > EMA10) with volume confirmation
            long_condition = (close[i] > high_max[i]) and \
                           (close[i] > ema_10_aligned[i]) and \
                           (volume[i] > 1.5 * vol_ma_20[i])
            
            # Short when price breaks below lower Donchian channel in downtrend (close < EMA10) with volume confirmation
            short_condition = (close[i] < low_min[i]) and \
                            (close[i] < ema_10_aligned[i]) and \
                            (volume[i] > 1.5 * vol_ma_20[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 10-day EMA (trend reversal)
            if close[i] < ema_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 10-day EMA (trend reversal)
            if close[i] > ema_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals