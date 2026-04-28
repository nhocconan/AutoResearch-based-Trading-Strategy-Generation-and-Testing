#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w EMA trend filter and volume confirmation.
# Enter long when 1d Bull Power > 0, price > 1w EMA34, and volume spike.
# Enter short when 1d Bear Power < 0, price < 1w EMA34, and volume spike.
# Uses discrete position sizing (0.25) to control drawdown. Target: 12-37 trades/year.
# Elder Ray measures bull/bear strength via EMA13, 1w EMA34 provides higher timeframe trend filter,
# volume confirmation ensures breakout conviction. Works in bull (strong buying) and bear (strong selling) markets.

name = "6h_ElderRay_1wEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for Elder Ray (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 on 1d close
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_1d = high_1d - ema13_1d  # >0 indicates bullish strength
    bear_power_1d = low_1d - ema13_1d   # <0 indicates bearish strength
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Align 1w EMA34 to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions with 1w EMA filter and volume confirmation
        long_entry = bull_power_aligned[i] > 0 and close[i] > ema34_1w_aligned[i] and volume_spike[i]
        short_entry = bear_power_aligned[i] < 0 and close[i] < ema34_1w_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Elder Ray signal or price crosses 1w EMA
        long_exit = bull_power_aligned[i] <= 0 or close[i] <= ema34_1w_aligned[i]
        short_exit = bear_power_aligned[i] >= 0 or close[i] >= ema34_1w_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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