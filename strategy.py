#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze + 1d Donchian breakout direction + volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves. We use the 1d Donchian
# channel to determine the breakout direction (long if price > 1d upper band, short if < lower band).
# Entry occurs when 6h Bollinger Bands expand (volatility increase) in the direction of the 1d trend,
# confirmed by volume spike. This captures breakouts from low-volatility regimes with trend alignment.
# Targets 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.

name = "6h_BollingerSqueeze_1dDonchian_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands on 6h: 20-period SMA, 2 std dev
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    bandwidth = (upper - lower) / sma  # Bollinger Band Width
    
    # Bollinger Band squeeze: bandwidth < 20-period percentile (low volatility)
    bandwidth_percentile = pd.Series(bandwidth).rolling(window=50, min_periods=20).quantile(0.2).values
    squeeze = bandwidth < bandwidth_percentile
    
    # Bollinger Band expansion: bandwidth > previous bandwidth (volatility increasing)
    bandwidth_expanding = bandwidth > np.roll(bandwidth, 1)
    bandwidth_expanding[0] = False
    
    # Get 1d data for Donchian breakout direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian Channel on 1d: 20-period high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_period = 20
    upper_1d = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_1d = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    close_1d = df_1d['close'].values
    
    # Determine breakout direction: long if close > upper_1d, short if close < lower_1d
    breakout_long = close_1d > upper_1d
    breakout_short = close_1d < lower_1d
    
    # Align 1d indicators to 6h
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    breakout_long_aligned = align_htf_to_ltf(prices, df_1d, breakout_long)
    breakout_short_aligned = align_htf_to_ltf(prices, df_1d, breakout_short)
    
    # Volume confirmation: current volume > 2.5x 20-period average (high threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for BB and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bandwidth[i]) or np.isnan(bandwidth_percentile[i]) or
            np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_width = bandwidth[i]
        bb_squeeze = squeeze[i]
        bb_expanding = bandwidth_expanding[i]
        upper_1d_val = upper_1d_aligned[i]
        lower_1d_val = lower_1d_aligned[i]
        breakout_long_val = breakout_long_aligned[i]
        breakout_short_val = breakout_short_aligned[i]
        vol_conf_val = vol_conf[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: BB squeeze ending, expanding, breakout long signal, volume confirmation
            if (not bb_squeeze) and bb_expanding and breakout_long_val and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: BB squeeze ending, expanding, breakout short signal, volume confirmation
            elif (not bb_squeeze) and bb_expanding and breakout_short_val and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: BB re-squeezes or breakout fails or opposite signal
            if bb_squeeze or not breakout_long_val or breakout_short_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: BB re-squeezes or breakout fails or opposite signal
            if bb_squeeze or not breakout_short_val or breakout_long_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals