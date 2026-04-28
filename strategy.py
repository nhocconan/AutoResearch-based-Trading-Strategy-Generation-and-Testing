#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d EMA34 trend + volume spike confirmation.
# Williams %R identifies overbought/oversold conditions: long when %R crosses above -80 from below,
# short when %R crosses below -20 from above. 1d EMA34 provides trend filter: long only when price > EMA34,
# short only when price < EMA34. Volume spike (>2.0x 20-bar average) confirms momentum.
# Target: 12-37 trades/year (50-150 total over 4 years). Discrete size 0.25 minimizes fee churn.
# Works in both bull and bear markets via trend filter preventing counter-trend entries.

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Align HTF indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for Williams %R and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals: long when crosses above -80 from below, short when crosses below -20 from above
        wr_long_signal = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_short_signal = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        # Trend filter: 1d EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Exit conditions: Williams %R reverts to opposite extreme or trend reversal
        wr_long_exit = williams_r[i] > -20  # Exit long when overbought
        wr_short_exit = williams_r[i] < -80  # Exit short when oversold
        trend_long_exit = close[i] < ema_34_1d_aligned[i]  # Exit long when price below EMA
        trend_short_exit = close[i] > ema_34_1d_aligned[i]  # Exit short when price above EMA
        
        # Handle entries and exits
        if wr_long_signal and price_above_ema and volume_spike[i] and position <= 0:
            signals[i] = 0.25
            position = 1
        elif wr_short_signal and price_below_ema and volume_spike[i] and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and (wr_long_exit or trend_long_exit)) or \
             (position == -1 and (wr_short_exit or trend_short_exit)):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals