# 12h_1D_Range_Breakout_With_Volume_Confirmation
# Hypothesis: Breakouts from daily trading ranges with volume confirmation capture directional moves in both bull and bear markets.
# Uses daily high/low as key support/resistance levels. Volume spike confirms institutional participation.
# Works in bull markets (breakouts to new highs) and bear markets (breakdowns to new lows).
# Timeframe: 12h balances trade frequency and signal quality, avoiding excessive churn.
# Volume filter prevents false breakouts in low-liquidity periods.
# Expect 15-30 trades/year per symbol, balancing opportunity with cost efficiency.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for key levels and volume context
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily range (high-low) and its 20-period average for volatility filter
    daily_range = daily['high'].values - daily['low'].values
    avg_daily_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    avg_daily_range_aligned = align_htf_to_ltf(prices, daily, avg_daily_range)
    
    # Volume spike detection: current volume > 1.5x 20-day average volume
    vol_ma_20d = pd.Series(daily['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, daily, vol_ma_20d)
    volume_threshold = 1.5 * vol_ma_20d_aligned
    volume_spike = volume > volume_threshold
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(avg_daily_range_aligned[i]) or np.isnan(vol_ma_20d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get today's daily high and low (aligned to current 12h bar)
        daily_high = align_htf_to_ltf(prices, daily, daily['high'].values)[i]
        daily_low = align_htf_to_ltf(prices, daily, daily['low'].values)[i]
        
        # Only trade when volatility is sufficient (avoid low volatility chop)
        if avg_daily_range_aligned[i] < 0.01 * close[i]:  # Less than 1% of price
            signals[i] = 0.0
            continue
            
        # Long: Price breaks above daily high with volume spike
        if close[i] > daily_high and volume_spike[i]:
            signals[i] = 0.25
        
        # Short: Price breaks below daily low with volume spike
        elif close[i] < daily_low and volume_spike[i]:
            signals[i] = -0.25
        
        # Exit: reverse signal when price returns to opposite daily level
        elif close[i] < daily_low and signals[i-1] > 0:
            signals[i] = 0.0
        elif close[i] > daily_high and signals[i-1] < 0:
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_1D_Range_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0