#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 Trend Filter and Volume Spike
# Long when price > Alligator Jaw AND price > 1w EMA50 (strong uptrend) AND volume spike
# Short when price < Alligator Jaw AND price < 1w EMA50 (strong downtrend) AND volume spike
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# We use Jaw as the main trend indicator (slowest, most reliable)
# 1w EMA50 provides multi-timeframe trend filter, reducing whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while capturing trends
# Works in bull (trend + breaks above Jaw) and bear (mean reversion below Jaw + volume confirmation)
# Timeframe: 1d (primary timeframe as required)

name = "1d_WilliamsAlligator_Jaw_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator components on 1d
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_1d = (high + low) / 2.0
    jaw_raw = smma(median_price_1d, 13)
    jaw_shifted = np.roll(jaw_raw, 8)  # Shift 8 bars forward
    jaw_shifted[:8] = np.nan  # First 8 values are invalid after shift
    
    # Align Jaw to 1d timeframe (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 1d (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(jaw_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Jaw AND price > 1w EMA50 (strong uptrend) AND volume spike
            if (close[i] > jaw_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND price < 1w EMA50 (strong downtrend) AND volume spike
            elif (close[i] < jaw_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Jaw OR closes below 1w EMA50
            if close[i] < jaw_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Jaw OR closes above 1w EMA50
            if close[i] > jaw_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals