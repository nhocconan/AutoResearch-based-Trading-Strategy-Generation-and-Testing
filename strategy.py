#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R with EMA trend filter and volume spike confirmation.
# Enter long when Williams %R crosses above -80 from below, price > 1d EMA34, and volume > 2x 20-bar average.
# Enter short when Williams %R crosses below -20 from above, price < 1d EMA34, and volume > 2x 20-bar average.
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 20-50 trades/year.
# Williams %R captures momentum extremes, EMA34 provides trend filter from higher timeframe, volume confirms breakout strength.
# Works in bull (momentum continuation) and bear (failed momentum reversals via exits) markets.

name = "4h_WilliamsR14_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    williams_r = np.full(n_1d, np.nan)
    
    for i in range(13, n_1d):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close_1d[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50.0
    
    # Calculate 1d EMA34
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R crossover conditions with EMA filter and volume confirmation
        long_condition = (williams_r_aligned[i] > -80 and 
                         williams_r_aligned[i-1] <= -80 and  # crossed above -80
                         close[i] > ema_34_1d_aligned[i] and 
                         volume_spike[i])
        short_condition = (williams_r_aligned[i] < -20 and 
                          williams_r_aligned[i-1] >= -20 and  # crossed below -20
                          close[i] < ema_34_1d_aligned[i] and 
                          volume_spike[i])
        
        # Exit conditions: opposite Williams %R level
        long_exit = williams_r_aligned[i] < -50  # exit long when crosses below -50
        short_exit = williams_r_aligned[i] > -50  # exit short when crosses above -50
        
        # Handle entries and exits
        if long_condition and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.30
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals