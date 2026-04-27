# 1d_Camarilla_Pivot_WeeklyEMA34_Trend_Volume
# Hypothesis: Daily Camarilla pivot levels (R1, S1) provide high-probability reversal points.
# In weekly uptrend (price > EMA34) with volume spike (>2x avg), go long at S1 bounce.
# In weekly downtrend (price < EMA34) with volume spike, go short at R1 rejection.
# Weekly trend filter avoids counter-trend trades in strong moves.
# Volume surge confirms institutional participation at pivot levels.
# Designed for 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.
# Works in bull/bear markets by following weekly trend and requiring volatility confirmation.

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 1d timeframe (wait for daily bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 34-period EMA on weekly close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price at S1 support AND weekly uptrend AND volume spike
        if (low[i] <= s1_1d_aligned[i] * 1.001 and  # Allow small tolerance for wicks
            close[i] > s1_1d_aligned[i] and         # Close above S1 (bounce confirmed)
            close[i] > ema34_1w_aligned[i] and      # Weekly uptrend
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price at R1 resistance AND weekly downtrend AND volume spike
        elif (high[i] >= r1_1d_aligned[i] * 0.999 and  # Allow small tolerance for wicks
              close[i] < r1_1d_aligned[i] and          # Close below R1 (rejection confirmed)
              close[i] < ema34_1w_aligned[i] and       # Weekly downtrend
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_Pivot_WeeklyEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0