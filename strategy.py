#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily volatility breakout with weekly trend filter.
# Long when price breaks above daily high of past 20 periods with weekly EMA alignment and volume confirmation.
# Short when price breaks below daily low of past 20 periods with weekly EMA alignment and volume confirmation.
# Exit when price returns to daily midpoint or weekly EMA slope changes direction.
# Uses daily range breakout for structure, weekly EMA for trend filter, volume for confirmation.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for volatility breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily high/low of past 20 periods
    daily_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    daily_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    daily_mid = (daily_high + daily_low) / 2
    
    # Load weekly data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_mid_aligned = align_htf_to_ltf(prices, df_1d, daily_mid)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need daily range and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or
            np.isnan(daily_mid_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA
        price_above_ema = close[i] > ema_1w_aligned[i]
        price_below_ema = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Look for breakouts
            # Long: price breaks above daily high AND price above weekly EMA
            if (close[i] > daily_high_aligned[i] and 
                price_above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below daily low AND price below weekly EMA
            elif (close[i] < daily_low_aligned[i] and 
                  price_below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to daily midpoint or price crosses below weekly EMA
            if (close[i] <= daily_mid_aligned[i] or 
                close[i] < ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to daily midpoint or price crosses above weekly EMA
            if (close[i] >= daily_mid_aligned[i] or 
                close[i] > ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_DailyRangeBreakout_WeeklyEMA_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0