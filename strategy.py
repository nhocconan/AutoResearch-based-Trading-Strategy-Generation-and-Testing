# 4H_1D_CAMARILLA_R1_S1_BREAKOUT_1DTREND_VOLUME_SPIKE
# Hypothesis: Use daily Camarilla levels (R1/S1) with 1d trend filter and volume spike confirmation on 4h chart.
# Enter long when price breaks above daily R1 with 1d uptrend and volume > 2x 20-period average.
# Enter short when price breaks below daily S1 with 1d downtrend and volume > 2x 20-period average.
# Exit when price reverses to opposite Camarilla level or trend changes.
# Tight entry conditions target 20-40 trades/year to minimize fee drag.
# Works in bull markets (trend continuation) and bear markets (mean reversion at extremes).

name = "4H_1D_CAMARILLA_R1_S1_BREAKOUT_1DTREND_VOLUME_SPIKE"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: R1, S1 based on previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    # Camarilla R1 = close + (range * 1.1/12)
    # Camarilla S1 = close - (range * 1.1/12)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Volume spike: current volume > 2x 20-period average (tighter filter)
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_avg * 2.0)
    
    # Align 1d indicators to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R1 + 1d uptrend + volume spike
            if close[i] > camarilla_r1_aligned[i] and trend_up_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S1 + 1d downtrend + volume spike
            elif close[i] < camarilla_s1_aligned[i] and not trend_up_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S1 (reversal) or trend changes to down
            if close[i] < camarilla_s1_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R1 (reversal) or trend changes to up
            if close[i] > camarilla_r1_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals