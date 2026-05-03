#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume spike
# Williams %R identifies overbought/oversold conditions. In ranging markets, fade extremes.
# In trending markets (price > EMA34), allow continuation breaks above/below key levels.
# Volume spike confirms participation. Designed for 6h timeframe to capture medium-term swings
# in both bull and bear markets. Discrete sizing 0.25 to manage drawdown and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    def calculate_williams_r(high, low, close, length):
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = calculate_williams_r(high, low, close, 14)
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        wr_val = wr[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr_val) or np.isnan(ema_trend):
            continue
            
        # Entry conditions
        # Long: Williams %R crosses above -80 (oversold) with volume spike and above 1d EMA34
        long_entry = (wr_val > -80) and (wr[i-1] <= -80) and vol_spike and (close[i] > ema_trend)
        # Short: Williams %R crosses below -20 (overbought) with volume spike and below 1d EMA34
        short_entry = (wr_val < -20) and (wr[i-1] >= -20) and vol_spike and (close[i] < ema_trend)
        
        # Exit conditions
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses below -50 (momentum loss) or close below EMA
            long_exit = (wr_val < -50) or (close[i] < ema_trend)
        elif position == -1:  # Short position
            # Exit when Williams %R crosses above -50 (momentum loss) or close above EMA
            short_exit = (wr_val > -50) or (close[i] > ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals