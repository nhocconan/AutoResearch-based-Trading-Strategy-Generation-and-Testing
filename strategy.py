#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Long when Alligator jaws (13-period SMMA) < teeth (8-period SMMA) < lips (5-period SMMA) AND close > 1d EMA34 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when Alligator jaws > teeth > lips AND close < 1d EMA34 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when Alligator lines re-cross (jaws-teeth-lips no longer aligned) OR price touches opposite Alligator line
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trending vs ranging markets via SMMA alignment
# 1d EMA34 provides strong trend filter between 12h and 1w for better regime adaptation
# Volume spike threshold increased to 2.0x to reduce false breakouts and lower trade frequency

name = "12h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's MA or RMA"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.empty_like(data)
    result[:] = np.nan
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator lines using SMMA (Smoothed Moving Average)
    # Jaws: 13-period SMMA of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    jaws = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams Alligator signals with trend and volume filters
            # Long: Jaws < Teeth < Lips (bullish alignment) AND uptrend AND volume spike
            if jaws[i] < teeth[i] and teeth[i] < lips[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) AND downtrend AND volume spike
            elif jaws[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines re-cross (bullish alignment broken) OR price touches jaws
            if not (jaws[i] < teeth[i] and teeth[i] < lips[i]) or close[i] <= jaws[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines re-cross (bearish alignment broken) OR price touches lips
            if not (jaws[i] > teeth[i] and teeth[i] > lips[i]) or close[i] >= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals