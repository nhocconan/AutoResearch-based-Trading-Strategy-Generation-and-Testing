#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Band squeeze + 1w Camarilla pivot breakout
# Bollinger Band squeeze (low volatility) on 1d precedes explosive moves
# Breakout confirmed when price closes outside 1w Camarilla R4/S4 levels
# Volume filter: current 6h volume > 1.5x 20-period average to avoid false breakouts
# Position size: fixed 0.25 (25%) to minimize fee churn and control drawdown
# Works in both bull and bear markets as it captures volatility expansion phases
# Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1d_bb_squeeze_1w_camarilla_breakout_v1"
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
    
    # Load 1d data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Bollinger Bands (20, 2.0)
    bb_ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma_20 + 2.0 * bb_std_20
    bb_lower = bb_ma_20 - 2.0 * bb_std_20
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width below 50th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.50).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Load 1w data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Align indicators to 6h timeframe
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    bb_ma_20_aligned = align_htf_to_ltf(prices, df_1d, bb_ma_20)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_squeeze_aligned[i]) or np.isnan(bb_ma_20_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on retracement to 1d BB middle or stop at 1w S4 breakdown
            if close[i] < bb_ma_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i]:  # Stop loss at 1w S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on retracement to 1d BB middle or stop at 1w R4 breakout
            if close[i] > bb_ma_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i]:  # Stop loss at 1w R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Bollinger Band squeeze breakout with volume confirmation
            # Long breakout: price closes above 1w R4 and above 1d BB upper
            # Short breakout: price closes below 1w S4 and below 1d BB lower
            if bb_squeeze_aligned[i] and volume_confirmed:
                if close[i] > r4_aligned[i] and close[i] > bb_upper_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i] and close[i] < bb_lower_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals