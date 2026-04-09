#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout + 1w EMA200 Trend Filter + Volume Spike
# - Primary signal: 6h price breaks above Camarilla R4 (bullish) or below S4 (bearish) from prior 1d
# - Trend filter: 1w EMA200 - price must be above EMA for longs, below for shorts (align with weekly trend)
# - Volume confirmation: 6h volume > 1.5 × 20-period median volume (ensure participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla breakouts capture strong moves, weekly EMA200 filter avoids counter-trend trades,
#   volume spike confirms institutional participation reducing false breakouts

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla levels (prior day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior 1d Camarilla levels
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Camarilla levels: R4 = prior_close + 1.5 * (prior_high - prior_low)
    #                 S4 = prior_close - 1.5 * (prior_high - prior_low)
    camarilla_range = prior_high - prior_low
    r4 = prior_close + 1.5 * camarilla_range
    s4 = prior_close - 1.5 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    # 1w EMA200 for trend direction
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Align 1w EMA200 to 6h timeframe (completed 1w bar only)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume regime: volume > 1.5 × 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below R4 OR closes below weekly EMA200
            if close[i] < r4_aligned[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above S4 OR closes above weekly EMA200
            if close[i] > s4_aligned[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and weekly trend filter
            # Long: price > R4 AND volume spike AND price above weekly EMA200
            if close[i] > r4_aligned[i] and volume_spike[i] and close[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price < S4 AND volume spike AND price below weekly EMA200
            elif close[i] < s4_aligned[i] and volume_spike[i] and close[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals