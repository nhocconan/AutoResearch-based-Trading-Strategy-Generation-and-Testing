#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily high/low from prior day for mean reversion with volume confirmation
# - Long when price touches prior day's low with volume confirmation and closes back above it
# - Short when price touches prior day's high with volume confirmation and closes back below it
# - Uses 1d EMA50 to only take trades in direction of daily trend (long above EMA50, short below)
# - Designed to work in ranging markets by fading extreme daily levels
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_PriorDayHL_MeanReversion_1dEMA50_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for prior day's high/low and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high and low (shifted by 1 to avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    
    # Align prior day's high/low to 4h timeframe
    prior_high_4h = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_4h = align_htf_to_ltf(prices, df_1d, prior_low)
    
    # 1d EMA50 for trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(prior_high_4h[i]) or np.isnan(prior_low_4h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price touches prior day's low and closes back above it with volume
            if low[i] <= prior_low_4h[i] * 1.001 and close[i] > prior_low_4h[i] and volume_filter[i]:
                # Only take long if above daily EMA50 (bullish bias)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short setup: price touches prior day's high and closes back below it with volume
            elif high[i] >= prior_high_4h[i] * 0.999 and close[i] < prior_high_4h[i] and volume_filter[i]:
                # Only take short if below daily EMA50 (bearish bias)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches prior day's high (target) or breaks below prior day's low (stop)
            if high[i] >= prior_high_4h[i] * 0.999:  # Take profit at prior day's high
                signals[i] = 0.0
                position = 0
            elif low[i] < prior_low_4h[i]:  # Stop loss if breaks below prior day's low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches prior day's low (target) or breaks above prior day's high (stop)
            if low[i] <= prior_low_4h[i] * 1.001:  # Take profit at prior day's low
                signals[i] = 0.0
                position = 0
            elif high[i] > prior_high_4h[i]:  # Stop loss if breaks above prior day's high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals