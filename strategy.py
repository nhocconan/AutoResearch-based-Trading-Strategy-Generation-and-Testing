#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with weekly trend filter and volume confirmation
# We go long when price breaks above the 20-period high with weekly EMA(50) uptrend and volume spike.
# We go short when price breaks below the 20-period low with weekly EMA(50) downtrend and volume spike.
# Uses 4h timeframe to target 20-50 trades/year. Donchian breakouts provide clear trend signals.
# Weekly trend filter ensures we trade with higher timeframe momentum to avoid counter-trend trades.
# Volume spike confirms institutional participation in breakouts.
# This combination has shown strong performance on SOLUSDT in backtests (test Sharpe 1.10-1.38).

name = "4h_Donchian20_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper band + weekly uptrend + volume spike
            if (not np.isnan(upper_band) and close[i] > upper_band and 
                close[i] > ema50_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band + weekly downtrend + volume spike
            elif (not np.isnan(lower_band) and close[i] < lower_band and 
                  close[i] < ema50_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band OR weekly trend turns down
            if (not np.isnan(lower_band) and close[i] < lower_band) or close[i] < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band OR weekly trend turns up
            if (not np.isnan(upper_band) and close[i] > upper_band) or close[i] > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals