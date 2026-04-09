#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
# Donchian captures breakouts; 1d Camarilla pivots (R3/S3 for mean reversion, R4/S4 for continuation)
# Volume ensures breakout authenticity; discrete sizing 0.25 limits drawdown
# Works in bull/bear: Camarilla levels adapt to volatility, breakouts work in both directions
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing

name = "6h_1d_camarilla_donchian_volume_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)  # Mean reversion sell
    camarilla_s3 = np.full(n, np.nan)  # Mean reversion buy
    camarilla_r4 = np.full(n, np.nan)  # Continuation sell
    camarilla_s4 = np.full(n, np.nan)  # Continuation buy
    
    # Camarilla formulas: based on previous day's range
    for i in range(n):
        if i < 24:  # Need at least 24 hours of 6h data to get previous day
            camarilla_high[i] = np.nan
            camarilla_low[i] = np.nan
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_r4[i] = np.nan
            camarilla_s4[i] = np.nan
        else:
            # Get previous day's OHLC (assuming 4x 6h bars per day)
            prev_day_start = max(0, i - 4)
            prev_day_end = i
            if prev_day_end - prev_day_start >= 4:
                # Use the last completed day's data
                day_high = np.max(high[prev_day_start:prev_day_end])
                day_low = np.min(low[prev_day_start:prev_day_end])
                day_close = close[prev_day_end - 1]  # Close of previous bar
                
                # Camarilla calculations
                range_val = day_high - day_low
                camarilla_high[i] = day_high
                camarilla_low[i] = day_low
                camarilla_r3[i] = day_close + range_val * 1.1 / 4
                camarilla_s3[i] = day_close - range_val * 1.1 / 4
                camarilla_r4[i] = day_close + range_val * 1.1 / 2
                camarilla_s4[i] = day_close - range_val * 1.1 / 2
            else:
                camarilla_high[i] = np.nan
                camarilla_low[i] = np.nan
                camarilla_r3[i] = np.nan
                camarilla_s3[i] = np.nan
                camarilla_r4[i] = np.nan
                camarilla_s4[i] = np.nan
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(avg_volume[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < Camarilla S3 (mean reversion) OR price > Camarilla R4 (continuation failed)
            if close[i] < donchian_low[i] or close[i] < camarilla_s3[i] or close[i] > camarilla_r4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > Camarilla R3 (mean reversion) OR price < Camarilla S4 (continuation failed)
            if close[i] > donchian_high[i] or close[i] > camarilla_r3[i] or close[i] < camarilla_s4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirmed:
                # Long entry: Donchian breakout above resistance with Camarilla support
                if close[i] > donchian_high[i] and close[i] > camarilla_s3[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Donchian breakdown below support with Camarilla resistance
                elif close[i] < donchian_low[i] and close[i] < camarilla_r3[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals