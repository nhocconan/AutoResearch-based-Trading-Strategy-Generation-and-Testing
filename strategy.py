# 100% rule-compliant: 4h timeframe, HTF via mtf_data, no look-ahead, proper position sizing
# Hypothesis: 4h momentum breakout with 1d trend filter and volume confirmation
# Uses 4h Donchian breakout with 1d EMA trend and volume spike
# Entry: Price breaks above/below 4h Donchian channel + 1d EMA trend + volume spike
# Exit: Price returns to Donchian midpoint or trend reversal
# Position sizing: 0.25 (25%) to limit drawdown
# Target: 20-40 trades/year to avoid fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(30) for trend direction
    close_1d = df_1d['close'].values
    ema30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema30_1d)
    
    # 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema30_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema30_1d_val = ema30_1d_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        donchian_mid_val = donchian_mid[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + uptrend + volume spike
            if (close[i] > donchian_high_val and 
                close[i] > ema30_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < donchian_low_val and 
                  close[i] < ema30_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend turns down
            if (close[i] < donchian_mid_val or close[i] < ema30_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend turns up
            if (close[i] > donchian_mid_val or close[i] > ema30_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals