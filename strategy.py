#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 Breakout with 4h EMA50 Trend Filter and Volume Spike
# Long when price breaks above R1 (4h) AND price > 4h EMA50 (strong uptrend) AND volume spike
# Short when price breaks below S1 (4h) AND price < 4h EMA50 (strong downtrend) AND volume spike
# R1/S1 are inner Camarilla levels (PP ± range*1.1/4) for high-probability breaks with lower false signals
# 4h EMA50 provides medium-term trend filter, reducing whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Target: 80-150 total trades over 4 years (20-38/year) to minimize fee drag while capturing trends
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 1h (primary timeframe as required)

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data ONCE before loop for Camarilla levels and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of completed 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift by 1 to use only completed 4h bar (look-ahead safety)
    high_4h_shifted = np.roll(high_4h, 1)
    low_4h_shifted = np.roll(low_4h, 1)
    close_4h_shifted = np.roll(close_4h, 1)
    
    # Calculate pivot point (PP) = (H+L+C)/3
    pp = (high_4h_shifted + low_4h_shifted + close_4h_shifted) / 3.0
    # Calculate range
    range_4h = high_4h_shifted - low_4h_shifted
    # Camarilla levels (R1/S1 = PP ± range*1.1/4)
    r1 = pp + (range_4h * 1.1 / 4.0)  # R1 = PP + range*1.1/4
    s1 = pp - (range_4h * 1.1 / 4.0)  # S1 = PP - range*1.1/4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation on 1h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND strong uptrend (price > 4h EMA50) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND strong downtrend (price < 4h EMA50) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below R1 OR closes below 4h EMA50
            if close[i] < r1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above S1 OR closes above 4h EMA50
            if close[i] > s1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals