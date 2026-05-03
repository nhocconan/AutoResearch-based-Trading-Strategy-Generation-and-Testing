#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA34 trend filter and volume confirmation.
# Long when 6h Williams %R < -80 (oversold) AND 1d close > 1d EMA34 (uptrend) AND 6h volume > 1.5x 20-period volume MA.
# Short when 6h Williams %R > -20 (overbought) AND 1d close < 1d EMA34 (downtrend) AND 6h volume > 1.5x 20-period volume MA.
# Exit when Williams %R returns to -50 (mean reversion center) or trend reverses.
# Uses 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Williams %R captures exhaustion moves, 1d EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading reversals in the direction of the 1d trend when volume confirms.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        williams_r_val = williams_r[i]
        
        # Volume spike condition: current 6h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 1.5)
        
        # Williams %R conditions
        oversold = williams_r_val < -80   # Oversold condition
        overbought = williams_r_val > -20  # Overbought condition
        mean_reversion = williams_r_val > -50  # Exit long when back above -50
        mean_reversion_short = williams_r_val < -50  # Exit short when back below -50
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Williams %R oversold AND 1d uptrend AND volume spike AND session
            if oversold and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND 1d downtrend AND volume spike AND session
            elif overbought and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR trend changes
            if mean_reversion or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR trend changes
            if mean_reversion_short or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals