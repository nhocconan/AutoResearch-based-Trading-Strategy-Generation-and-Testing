# 1D_Camarilla_Pivot_R1_S1_Breakout_1W_EMA50_Trend
# Hypothesis: Daily Camarilla pivot breakouts with weekly EMA50 trend filter and volume confirmation
# work in both bull and bear markets by capturing institutional interest at key levels while
# avoiding counter-trend trades. Weekly EMA50 ensures we only trade in the direction of
# the longer-term trend, reducing whipsaws. Volume confirmation filters breakouts with low
# participation. Designed for low trade frequency (target: 15-25 trades/year) to minimize
# fee drag. Uses 1d timeframe with 1h trend filter for multi-timeframe confluence.

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
    
    # Load 1d data for pivot points (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    range_1d = high_1d - low_1d
    close_prev = close_1d
    h5 = close_prev + 1.1 * range_1d / 2  # Resistance level 5
    h4 = close_prev + 1.1 * range_1d / 4  # Resistance level 4
    h3 = close_prev + 1.1 * range_1d / 6  # Resistance level 3
    l3 = close_prev - 1.1 * range_1d / 6  # Support level 3
    l4 = close_prev - 1.1 * range_1d / 4  # Support level 4
    l5 = close_prev - 1.1 * range_1d / 2  # Support level 5
    
    # Align Camarilla levels to 1d timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1d, h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, l5)
    
    # Load 1h data for EMA50 trend filter (ONCE before loop)
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1h close
    close_1h = df_1h['close'].values
    ema_50 = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(h5_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(l5_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above H4 (strong resistance) + volume spike + above weekly EMA50
            if close[i] > h4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 (strong support) + volume spike + below weekly EMA50
            elif close[i] < l4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite H3/L3 level (partial retracement)
            if position == 1:
                # Exit long: Price closes below H3
                if close[i] < h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above L3
                if close[i] > l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_Camarilla_Pivot_R1_S1_Breakout_1W_EMA50_Trend"
timeframe = "1d"
leverage = 1.0